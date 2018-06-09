package de.bbuzz18.nsf.streaming.functions;

import java.util.Collection;
import java.util.LinkedList;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.Similarity;

/**
 *
 */
public class SearcherFactoryFunction implements MapFunction<IndexReader, Collection<IndexSearcher>> {


  private Collection<Similarity> similarities;

  public SearcherFactoryFunction(Collection<Similarity> similarities) {
    this.similarities = similarities;
  }

  @Override
  public Collection<IndexSearcher> map(IndexReader value) throws Exception {
    Collection<IndexSearcher> searchers = new LinkedList<>();
    for (Similarity s : similarities) {
      IndexSearcher indexSearcher = new IndexSearcher(value);
      indexSearcher.setSimilarity(s);
      searchers.add(indexSearcher);
    }
    return searchers;
  }
}
