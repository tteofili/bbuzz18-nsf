package de.bbuzz18.nsf.streaming.functions;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

/**
 *
 */
public class MultiRetrieverFunction implements MapFunction<Collection<IndexSearcher>, Map<String, String[]>> {
  private SimpleQueryParser simpleQueryParser = new SimpleQueryParser(new StandardAnalyzer(), "text");

  @Override
  public Map<String, String[]> map(Collection<IndexSearcher> value) throws Exception {
    Map<String, String[]> results = new HashMap<>();
    int topK = 5;
    for (IndexSearcher searcher : value) {
      Query query = simpleQueryParser.parse("berlin buzzwords talk about neural search frontier");
      TopDocs topDocs = searcher.search(query, topK);
      String[] stringResults = new String[topK];
      int i = 0;
      for (ScoreDoc sd : topDocs.scoreDocs) {
        stringResults[i] = searcher.doc(sd.doc).getField("title").stringValue();
        i++;
      }
      results.put(searcher.getSimilarity(false).toString(), stringResults);
    }
    return results;
  }
}
