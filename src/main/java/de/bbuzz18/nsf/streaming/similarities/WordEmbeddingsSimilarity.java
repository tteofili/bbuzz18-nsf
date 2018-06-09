package de.bbuzz18.nsf.streaming.similarities;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import lucene4ir.Lucene4IRConstants;
import lucene4ir.similarity.VectorizeUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 */
public class WordEmbeddingsSimilarity extends Similarity {

  public enum Smoothing {
    MEAN,
    IDF,
    TF,
    TF_IDF
  }

  private final Word2Vec word2Vec;
  private final String fieldName;
  private final Smoothing smoothing;

  public WordEmbeddingsSimilarity(Word2Vec word2Vec, String fieldName, Smoothing smoothing) {
    this.word2Vec = word2Vec;
    this.fieldName = fieldName;
    this.smoothing = smoothing;
  }

  public WordEmbeddingsSimilarity(Word2Vec word2Vec, String fieldName) {
    this.word2Vec = word2Vec;
    this.fieldName = fieldName;
    this.smoothing = Smoothing.MEAN;
  }

  @Override
  public long computeNorm(FieldInvertState state) {
    return 1l;
  }

  @Override
  public SimWeight computeWeight(float boost, CollectionStatistics collectionStats,
                                 TermStatistics... termStats) {
    return new EmbeddingsSimWeight(boost, collectionStats, termStats);
  }

  @Override
  public SimScorer simScorer(SimWeight weight, LeafReaderContext context) throws IOException {
    return new EmbeddingsSimScorer(weight, context);
  }

  private class EmbeddingsSimScorer extends SimScorer {
    private final EmbeddingsSimWeight weight;
    private final LeafReaderContext context;
    private Terms fieldTerms;
    private LeafReader reader;

    public EmbeddingsSimScorer(SimWeight weight, LeafReaderContext context) {
      this.weight = (EmbeddingsSimWeight) weight;
      this.context = context;
      this.reader = context.reader();
    }

    @Override
    public String toString() {
      return "EmbeddingsSimScorer{" +
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
        INDArray denseDocumentVector;
        Document document = reader.document(doc);
        BytesRef bytesRef;
        if (document != null && (bytesRef = document.getBinaryValue("wv")) != null) {
          denseDocumentVector = Nd4j.fromByteArray(bytesRef.bytes);
        } else {
          denseDocumentVector = toDenseAverageVector(reader.getTermVector(doc, fieldName), reader.numDocs(), word2Vec, smoothing);
        }
        return (float) Transforms.cosineSim(denseQueryVector, denseDocumentVector);
      } catch (IOException e) {
        return 0f;
      }
    }

    private INDArray getQueryVector() throws IOException {
      List<String> queryTerms = new LinkedList<>();
      for (TermStatistics termStats : weight.termStats) {
        queryTerms.add(termStats.term().utf8ToString());
      }
      return VectorizeUtils.averageWordVectors(queryTerms, word2Vec.getLookupTable());
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

  private class EmbeddingsSimWeight extends SimWeight {
    private final float boost;
    private final CollectionStatistics collectionStats;
    private final TermStatistics[] termStats;

    public EmbeddingsSimWeight(float boost, CollectionStatistics collectionStats, TermStatistics[] termStats) {
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

  private static INDArray toDenseAverageVector(Terms docTerms, double n, Word2Vec word2Vec,
                                               WordEmbeddingsSimilarity.Smoothing smoothing) throws IOException {
    INDArray vector = Nd4j.zeros(word2Vec.getLayerSize());
    if (docTerms != null) {
      TermsEnum docTermsEnum = docTerms.iterator();
      BytesRef term;
      while ((term = docTermsEnum.next()) != null) {
        INDArray wordVector = word2Vec.getLookupTable().vector(term.utf8ToString());
        if (wordVector != null) {
          long termFreq = docTermsEnum.totalTermFreq();
          int docFreq = docTermsEnum.docFreq();

          double smooth;
          switch (smoothing) {
            case MEAN:
              smooth = docTerms.size();
              break;
            case TF:
              smooth = termFreq;
              break;
            case IDF:
              smooth = docFreq;
              break;
            case TF_IDF:
              smooth = VectorizeUtils.tfIdf(n, termFreq, docFreq);
              break;
            default:
              smooth = docTerms.size();
          }
          vector.addi(wordVector.div(smooth));
        }
      }
    }
    return vector;
  }
}
