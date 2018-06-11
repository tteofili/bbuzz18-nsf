package de.bbuzz18.nsf.streaming;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import de.bbuzz18.nsf.streaming.functions.Utils;
import de.bbuzz18.nsf.streaming.functions.index.similarities.ParagraphVectorsSimilarity;
import de.bbuzz18.nsf.streaming.functions.index.similarities.WordEmbeddingsSimilarity;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 */
public class OutputApp {

  private static final Logger log = LoggerFactory.getLogger(OutputApp.class);

  public static void main(String[] args) throws Exception {
    Path indexPath = Utils.getIndexPath();
    ParagraphVectors paragraphVectors = Utils.fetchVectors(indexPath);

    String fieldName = "text";

    Map<String, IndexSearcher> searchers = new HashMap<>();
    FSDirectory directory = FSDirectory.open(indexPath);

    IndexReader reader1 = DirectoryReader.open(directory);
    IndexSearcher classic = new IndexSearcher(reader1);
    classic.setSimilarity(new ClassicSimilarity());
    searchers.put("classic", classic);

    IndexReader reader2 = DirectoryReader.open(directory);
    IndexSearcher bm25 = new IndexSearcher(reader2);
    bm25.setSimilarity(new BM25Similarity());
    searchers.put("bm25", bm25);

    IndexReader reader3 = DirectoryReader.open(directory);
    IndexSearcher pv = new IndexSearcher(reader3);
    pv.setSimilarity(new ParagraphVectorsSimilarity(paragraphVectors, fieldName));
    searchers.put("document embedding ranking", pv);

    IndexReader reader4 = DirectoryReader.open(directory);
    IndexSearcher lmd = new IndexSearcher(reader4);
    lmd.setSimilarity(new LMDirichletSimilarity());
    searchers.put("language model dirichlet", lmd);

    IndexReader reader5 = DirectoryReader.open(directory);
//    IndexSearcher wv = new IndexSearcher(reader5);
//    wv.setSimilarity(new WordEmbeddingsSimilarity(paragraphVectors, fieldName));
//    searchers.put("word embedding ranking", wv);


    Map<String, String[]> results = new HashMap<>();
    int topK = 3;
    QueryParser simpleQueryParser = new QueryParser(fieldName, new StandardAnalyzer());
    String queryText = "\"berlin buzzwords\" \"relevant search\" \"deep learning\" \"neural search\"";
    for (Map.Entry<String, IndexSearcher> entry : searchers.entrySet()) {
      Query query = simpleQueryParser.parse(queryText);
      log.debug("running query '{}' for {}", query.toString(), entry.getKey());
      IndexSearcher searcher = entry.getValue();
      TopDocs topDocs = searcher.search(query, topK);
      String[] stringResults = new String[topK];
      int i = 0;
      for (ScoreDoc sd : topDocs.scoreDocs) {
        Document doc = searcher.doc(sd.doc);
        IndexableField text = doc.getField(fieldName);
        if (text != null) {
          stringResults[i] = text.stringValue().replaceAll(",","");
        }
        i++;
      }
      results.put(entry.getKey() + " (" + topDocs.getMaxScore() + ")", stringResults);

    }

    reader1.close();
    reader2.close();
    reader3.close();
    reader4.close();
    reader5.close();
    directory.close();
    for (Map.Entry<String, String[]> entry : results.entrySet()) {
      log.info("{}:{}", entry.getKey(), Arrays.toString(entry.getValue()));
    }
  }
}
