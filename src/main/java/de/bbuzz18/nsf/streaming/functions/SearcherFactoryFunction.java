package de.bbuzz18.nsf.streaming.functions;

import java.util.Collection;
import java.util.LinkedList;

import de.bbuzz18.nsf.streaming.similarities.ParagraphVectorsSimilarity;
import de.bbuzz18.nsf.streaming.similarities.WordEmbeddingsSimilarity;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;

/**
 *
 */
public class SearcherFactoryFunction implements MapFunction<IndexReader, Collection<IndexSearcher>> {

  private ParagraphVectors paragraphVectors;

  public SearcherFactoryFunction(ParagraphVectors paragraphVectors) {
    this.paragraphVectors = paragraphVectors;
  }

  @Override
  public Collection<IndexSearcher> map(IndexReader value) throws Exception {
    Collection<IndexSearcher> searchers = new LinkedList<>();
    IndexSearcher indexSearcher = new IndexSearcher(value);
    indexSearcher.setSimilarity(new ClassicSimilarity());
    searchers.add(indexSearcher);
    indexSearcher = new IndexSearcher(value);
    indexSearcher.setSimilarity(new BM25Similarity());
    searchers.add(indexSearcher);
    indexSearcher = new IndexSearcher(value);
    indexSearcher.setSimilarity(new LMDirichletSimilarity());
    searchers.add(indexSearcher);
    indexSearcher = new IndexSearcher(value);
    indexSearcher.setSimilarity(new ParagraphVectorsSimilarity(paragraphVectors, "text"));
    searchers.add(indexSearcher);
    indexSearcher = new IndexSearcher(value);
    indexSearcher.setSimilarity(new WordEmbeddingsSimilarity(paragraphVectors, "text"));
    searchers.add(indexSearcher);
    return searchers;
  }
}
