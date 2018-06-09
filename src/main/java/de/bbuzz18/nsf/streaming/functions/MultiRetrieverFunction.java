package de.bbuzz18.nsf.streaming.functions;

import java.nio.file.Path;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import de.bbuzz18.nsf.streaming.similarities.ParagraphVectorsSimilarity;
import de.bbuzz18.nsf.streaming.similarities.WordEmbeddingsSimilarity;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.SearcherManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;

/**
 *
 */
public class MultiRetrieverFunction implements MapFunction<Path, Map<String, String[]>> {

  private final ParagraphVectors paragraphVectors;

  public MultiRetrieverFunction(ParagraphVectors paragraphVectors) {
    this.paragraphVectors = paragraphVectors;
  }

  @Override
  public Map<String, String[]> map(Path path) throws Exception {

    Map<String, IndexSearcher> searchers = new HashMap<>();
    FSDirectory directory = FSDirectory.open(path);

    SearcherManager manager = new SearcherManager(directory,null);
    IndexSearcher classic = manager.acquire();
    classic.setSimilarity(new ClassicSimilarity());
    searchers.put("classic",classic);

    IndexSearcher bm25 = manager.acquire();
    bm25.setSimilarity(new BM25Similarity());
    searchers.put("bm25", bm25);

    IndexSearcher pv = manager.acquire();
    pv.setSimilarity(new ParagraphVectorsSimilarity(paragraphVectors, "text"));
    searchers.put("pv", pv);

    Map<String, String[]> results = new HashMap<>();
    int topK = 3;
    SimpleQueryParser simpleQueryParser = new SimpleQueryParser(new StandardAnalyzer(), "text");
    for (Map.Entry<String,IndexSearcher> entry : searchers.entrySet()) {
      Query query = simpleQueryParser.parse("berlin buzzwords talk about neural search frontier");
      IndexSearcher searcher = entry.getValue();
      TopDocs topDocs = searcher.search(query, topK);
      String[] stringResults = new String[topK];
      int i = 0;
      for (ScoreDoc sd : topDocs.scoreDocs) {
        Document doc = searcher.doc(sd.doc);
        IndexableField text = doc.getField("text");
        if (text != null) {
          stringResults[i] = text.stringValue();
        }
        i++;
      }
      results.put(entry.getKey(), stringResults);
    }
    for (IndexSearcher s : searchers.values()) {
      manager.release(s);
    }
    manager.close();
    return results;
  }
}
