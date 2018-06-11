package de.bbuzz18.nsf.streaming.functions;

import java.util.Collection;

import de.bbuzz18.nsf.streaming.Tweet;
import de.bbuzz18.nsf.streaming.functions.index.CustomWriter;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.util.Collector;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 */
public class ModelAndIndexUpdateFunction implements AllWindowFunction<Tweet, Long, GlobalWindow> {

  private final Logger log = LoggerFactory.getLogger(getClass());
//  private final LabelsSource labelsSource;
//
//  public ModelAndIndexUpdateFunction(LabelsSource labelsSource) {
//    this.labelsSource = labelsSource;
//  }

  @Override
  public void apply(GlobalWindow globalWindow, Iterable<Tweet> iterable, Collector<Long> collector) throws Exception {
//    ParagraphVectors paragraphVectors = WordVectorSerializer.readParagraphVectors("pv.zip");
//    for (String l : labelsSource.getLabels()) {
//      paragraphVectors.getLabelsSource().storeLabel(l);
//    }
//    paragraphVectors.setTokenizerFactory(new DefaultTokenizerFactory());
    ParagraphVectors paragraphVectors = Utils.fetchVectors();
//    Map<String, INDArray> newPVs = new HashMap<>();
    // for each tweet
    CustomWriter writer = new CustomWriter();
    for (Tweet tweet : iterable) {

      // create lucene doc
      Document document = new Document();
      document.add(new StringField("id", tweet.getId(), Field.Store.YES));
      document.add(new StringField("lang", tweet.getLanguage(), Field.Store.YES));
      document.add(new StringField("user", tweet.getUser(), Field.Store.YES));
      document.add(new TextField("text", tweet.getText(), Field.Store.YES));

      // extract and index tweet embeddings
      if (tweet.getText() != null && tweet.getText().trim().length() > 0) {
        try {
          INDArray paragraphVector = paragraphVectors.inferVector(tweet.getText());
          log.debug("learned pv for {}", tweet.getId());
//          newPVs.put(tweet.getId(), paragraphVector);

          // ingest vectors for current tweet
          document.add(new BinaryDocValuesField("pv", new BytesRef(paragraphVector.data().asBytes())));
          INDArray averageWordVectors = averageWordVectors(paragraphVectors.getTokenizerFactory().create(tweet.getText()).getTokens(), paragraphVectors.lookupTable());
          document.add(new BinaryDocValuesField("wv", new BytesRef(averageWordVectors.data().asBytes())));

        } catch (Exception e) {
          System.err.println(tweet.getText() +" -> "+e.getLocalizedMessage());
        }
      }
      writer.addDocument(document);
    }
    long commit = writer.commit();
    log.info("Lucene index updated ({})", commit);

//    DirectoryReader reader = DirectoryReader.open(writer);
//    IndexSearcher searcher = new IndexSearcher(reader);
//    int i = 0;
//    for (Tweet t : iterable) {
//      if (newPVs.containsKey(t.getId())) {
//        Query q = new TermQuery(new Term("id", t.getId()));
//        TopDocs topDocs = searcher.search(q, 1);
//        for (ScoreDoc sd : topDocs.scoreDocs) {
//          String label = "doc_" + sd.doc;
//          paragraphVectors.getLookupTable().putVector(label, newPVs.get(t.getId()));
//          paragraphVectors.getLabelsSource().storeLabel(label);
//          i++;
//        }
//      }
//    }
//    log.info("{} embeddings updated", i);
//
//    WordVectorSerializer.writeParagraphVectors(paragraphVectors, "pv.zip");
//    Models models = new Models(writer.getPath(), paragraphVectors, paragraphVectors.getLabelsSource());

//    reader.close();
    writer.close();

    collector.collect(commit);
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
