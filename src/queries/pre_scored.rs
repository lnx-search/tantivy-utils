use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};

use tantivy::postings::SegmentPostings;
use tantivy::query::{Explanation, Query, Scorer, Weight};
use tantivy::schema::IndexRecordOption;
use tantivy::{DocId, DocSet, Score, Searcher, SegmentReader, Term};

#[derive(Clone)]
pub struct PreScoredQuery {
    term: Term,
    score: f32,
    index_record_option: IndexRecordOption,
}

impl Debug for PreScoredQuery {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PreScoredQuery(term={:?}, score={})",
            self.term, self.score
        )
    }
}

impl PreScoredQuery {
    pub fn new(
        term: Term,
        score: Score,
        index_record_option: IndexRecordOption,
    ) -> Self {
        Self {
            term,
            score,
            index_record_option,
        }
    }

    pub fn specialized_weight(
        &self,
        searcher: &Searcher,
        scoring_enabled: bool,
    ) -> tantivy::Result<PreScoredWeight> {
        let field_entry = searcher.schema().get_field_entry(self.term.field());
        if !field_entry.is_indexed() {
            let error_msg = format!("Field {:?} is not indexed.", field_entry.name());
            return Err(tantivy::TantivyError::SchemaError(error_msg));
        }

        let index_record_option = if scoring_enabled {
            self.index_record_option
        } else {
            IndexRecordOption::Basic
        };

        Ok(PreScoredWeight {
            term: self.term.clone(),
            score: self.score,
            index_record_option,
        })
    }
}

impl Query for PreScoredQuery {
    fn weight(
        &self,
        searcher: &Searcher,
        scoring_enabled: bool,
    ) -> tantivy::Result<Box<dyn Weight>> {
        Ok(Box::new(
            self.specialized_weight(searcher, scoring_enabled)?,
        ))
    }

    fn query_terms(&self, terms: &mut BTreeMap<Term, bool>) {
        terms.insert(self.term.clone(), false);
    }
}

pub struct PreScoredWeight {
    term: Term,
    score: Score,
    index_record_option: IndexRecordOption,
}

impl Weight for PreScoredWeight {
    fn scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> tantivy::Result<Box<dyn Scorer>> {
        Ok(Box::new(self.specialized_scorer(reader, boost)?))
    }

    fn explain(
        &self,
        reader: &SegmentReader,
        doc: DocId,
    ) -> tantivy::Result<Explanation> {
        let mut scorer = self.specialized_scorer(reader, 1.0)?;
        if scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(tantivy::TantivyError::InvalidArgument(format!(
                "Document #({}) does not match",
                doc
            )));
        }

        let mut explanation = Explanation::new(
            "Pre-scored query with a constant score for a given term.",
            self.score,
        );
        explanation.add_context(format!("Term={:?}", self.term));
        Ok(explanation)
    }

    fn count(&self, reader: &SegmentReader) -> tantivy::Result<u32> {
        if let Some(alive_bitset) = reader.alive_bitset() {
            Ok(self.scorer(reader, 1.0)?.count(alive_bitset))
        } else {
            let field = self.term.field();
            let inv_index = reader.inverted_index(field)?;
            let term_info = inv_index.get_term_info(&self.term)?;
            Ok(term_info.map(|term_info| term_info.doc_freq).unwrap_or(0))
        }
    }
}

impl PreScoredWeight {
    pub fn term(&self) -> &Term {
        &self.term
    }

    pub(crate) fn specialized_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> tantivy::Result<TermScorer> {
        let field = self.term.field();
        let inverted_index = reader.inverted_index(field)?;
        let postings_opt: Option<SegmentPostings> =
            inverted_index.read_postings(&self.term, self.index_record_option)?;

        if let Some(segment_postings) = postings_opt {
            Ok(TermScorer::new(segment_postings, self.score * boost))
        } else {
            Ok(TermScorer::new(
                SegmentPostings::empty(),
                self.score * boost,
            ))
        }
    }
}

pub struct TermScorer {
    postings: SegmentPostings,
    score: Score,
}

impl TermScorer {
    pub fn new(postings: SegmentPostings, score: Score) -> Self {
        Self { postings, score }
    }
}

impl DocSet for TermScorer {
    fn advance(&mut self) -> DocId {
        self.postings.advance()
    }

    fn doc(&self) -> DocId {
        self.postings.doc()
    }

    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }
}

impl Scorer for TermScorer {
    fn score(&mut self) -> Score {
        self.score
    }
}

#[cfg(test)]
mod tests {
    use tantivy::collector::TopDocs;
    use tantivy::merge_policy::NoMergePolicy;
    use tantivy::schema::{Schema, TEXT};
    use tantivy::{doc, Index};

    use super::*;

    #[test]
    fn test_pre_scored_query() -> tantivy::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;

        use rand::Rng;
        let mut rng = rand::thread_rng();
        writer.set_merge_policy(Box::new(NoMergePolicy));

        for _ in 0..3_000 {
            let term_freq = rng.gen_range(1..10000);
            let words: Vec<&str> = std::iter::repeat("bbbb").take(term_freq).collect();
            let text = words.join(" ");
            writer.add_document(doc!(text_field=>text))?;
        }
        writer.commit()?;

        let query = PreScoredQuery::new(
            Term::from_field_text(text_field, "bbbb"),
            1.0,
            IndexRecordOption::WithFreqs,
        );

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let collector = TopDocs::with_limit(1);
        let top = searcher.search(&query, &collector)?;

        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 1.0);

        Ok(())
    }
}
