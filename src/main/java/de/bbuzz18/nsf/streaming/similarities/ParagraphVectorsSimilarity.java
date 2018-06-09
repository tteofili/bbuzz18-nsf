package de.bbuzz18.nsf.streaming.similarities;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * a Lucene {@link Similarity} based on {@link ParagraphVectors}
 */
public class ParagraphVectorsSimilarity extends Similarity {

  private final ParagraphVectors paragraphVectors;
  private final String fieldName;

  public ParagraphVectorsSimilarity(ParagraphVectors paragraphVectors, String fieldName) {
    this.paragraphVectors = paragraphVectors;
    this.fieldName = fieldName;
  }

  @Override
  public long computeNorm(FieldInvertState state) {
    return 1l;
  }

  @Override
  public SimWeight computeWeight(float boost, CollectionStatistics collectionStats,
                                 TermStatistics... termStats) {
    return new PVSimWeight(boost, collectionStats, termStats);
  }

  @Override
  public SimScorer simScorer(SimWeight weight, LeafReaderContext context) throws IOException {
    return new PVSimScorer(weight, context);
  }

  private class PVSimScorer extends SimScorer {
    private final PVSimWeight weight;
    private final LeafReaderContext context;
    private Terms fieldTerms;
    private LeafReader reader;

    public PVSimScorer(SimWeight weight, LeafReaderContext context) {
      this.weight = (PVSimWeight) weight;
      this.context = context;
      this.reader = context.reader();
    }

    @Override
    public String toString() {
      return "PVSimScore{" +
          "weight=" + weight +
          ", context=" + context +
          ", fieldTerms=" + fieldTerms +
          ", reader=" + reader +
          '}';
    }

    @Override
    public float score(int doc, float freq) {
      try {
        INDArray denseQueryVector = getQueryVector();
        int adoc = context.docBase + doc;
        String label = "doc_" + adoc;
        INDArray documentParagraphVector = paragraphVectors.getLookupTable().vector(label);
        if (documentParagraphVector == null) {
          LabelledDocument document = new LabelledDocument();
          document.setLabels(Collections.singletonList(label));
          document.setContent(reader.document(adoc).getField(fieldName).stringValue());
          if (paragraphVectors.getTokenizerFactory() == null) {
            paragraphVectors.setTokenizerFactory(new DefaultTokenizerFactory());
          }
          documentParagraphVector = paragraphVectors.inferVector(document);
        }
        return (float) Transforms.cosineSim(denseQueryVector, documentParagraphVector);
      } catch (IOException e) {
        return 0f;
      }
    }

    private INDArray getQueryVector() throws IOException {
      StringBuilder q = new StringBuilder();
      for (TermStatistics termStats : weight.termStats) {
        if (q.length() > 0) {
          q.append(' ');
        }
        String str = termStats.term().utf8ToString();
        q.append(str);
      }
      INDArray indArray;
      String text = q.toString();
      try {
        if (paragraphVectors.getTokenizerFactory() == null) {
          paragraphVectors.setTokenizerFactory(new DefaultTokenizerFactory());
        }
        indArray = paragraphVectors.inferVector(text);
      } catch (ND4JIllegalStateException ne) {
        indArray = paragraphVectors.getLookupTable().vector(text);
      }
      return indArray;
    }

    @Override
    public float computeSlopFactor(int distance) {
      return 1;
    }

    @Override
    public float computePayloadFactor(int doc, int start, int end, BytesRef payload) {
      return 1;
    }
  }

  private class PVSimWeight extends SimWeight {
    private final float boost;
    private final CollectionStatistics collectionStats;
    private final TermStatistics[] termStats;

    public PVSimWeight(float boost, CollectionStatistics collectionStats, TermStatistics[] termStats) {
      this.boost = boost;
      this.collectionStats = collectionStats;
      this.termStats = termStats;
    }

    @Override
    public String toString() {
      return "EmbeddingsSimWeight{" +
          "boost=" + boost +
          ", collectionStats=" + collectionStats +
          ", termStats=" + Arrays.toString(termStats) +
          '}';
    }
  }
}
