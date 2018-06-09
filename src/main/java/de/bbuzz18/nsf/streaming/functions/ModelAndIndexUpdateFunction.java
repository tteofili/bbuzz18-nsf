package de.bbuzz18.nsf.streaming.functions;

import java.util.Collection;

import de.bbuzz18.nsf.streaming.CustomWriter;
import de.bbuzz18.nsf.streaming.Tweet;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.util.Collector;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 */
public class ModelAndIndexUpdateFunction implements AllWindowFunction<Tweet, IndexReader, GlobalWindow> {

  private final ParagraphVectors paragraphVectors;

  public ModelAndIndexUpdateFunction(ParagraphVectors paragraphVectors) {
    this.paragraphVectors = paragraphVectors;
  }

  @Override
  public void apply(GlobalWindow globalWindow, Iterable<Tweet> iterable, Collector<IndexReader> collector) throws Exception {
    // for each tweet
    IndexWriter writer = new CustomWriter();
    for (Tweet tweet : iterable) {

      // create lucene doc
      Document document = new Document();
      document.add(new StringField("id", tweet.getId(), Field.Store.YES));
      document.add(new StringField("lang", tweet.getLanguage(), Field.Store.YES));
      document.add(new StringField("user", tweet.getUser(), Field.Store.YES));
      document.add(new TextField("text", tweet.getText(), Field.Store.YES));

      // update models with current tweet
      if (tweet.getText() != null && tweet.getText().trim().length() > 0) {
        paragraphVectors.setTokenizerFactory(new DefaultTokenizerFactory());
        try {
          INDArray paragraphVector = paragraphVectors.inferVector(tweet.getText());

          // ingest vectors for current tweet
          document.add(new BinaryDocValuesField("pv", new BytesRef(paragraphVector.data().asBytes())));
          INDArray averageWordVectors = averageWordVectors(paragraphVectors.getTokenizerFactory().create(tweet.getText()).getTokens(), paragraphVectors.lookupTable());
          document.add(new BinaryDocValuesField("vector", new BytesRef(averageWordVectors.data().asBytes())));

        } catch (Exception e) {
          System.err.println(tweet.getText() +" -> "+e.getLocalizedMessage());
        }
      }
      writer.addDocument(document);
    }
    writer.commit();
    collector.collect(DirectoryReader.open(writer));
  }

  private static INDArray averageWordVectors(Collection<String> words, WeightLookupTable lookupTable) {
    INDArray denseDocumentVector = Nd4j.zeros(words.size(), lookupTable.layerSize());
    int i = 0;
    for (String w : words) {
      INDArray vector = lookupTable.vector(w);
      if (vector == null) {
        vector = lookupTable.vector("UNK");
      }
      denseDocumentVector.putRow(i, vector);
      i++;
    }
    return denseDocumentVector.mean(0);
  }
}
