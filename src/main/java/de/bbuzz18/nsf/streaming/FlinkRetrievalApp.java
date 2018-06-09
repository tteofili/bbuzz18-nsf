package de.bbuzz18.nsf.streaming;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Map;
import java.util.Properties;

import de.bbuzz18.nsf.streaming.functions.CommitFunction;
import de.bbuzz18.nsf.streaming.functions.IndexFunction;
import de.bbuzz18.nsf.streaming.functions.ModelUpdateFunction;
import de.bbuzz18.nsf.streaming.functions.MultiRetrieverFunction;
import de.bbuzz18.nsf.streaming.functions.ResultTransformer;
import de.bbuzz18.nsf.streaming.functions.SearcherFactoryFunction;
import de.bbuzz18.nsf.streaming.functions.TupleEvictorFunction;
import de.bbuzz18.nsf.streaming.functions.TweetJsonConverter;
import lucene4ir.similarity.ParagraphVectorsSimilarity;
import lucene4ir.similarity.WordEmbeddingsSimilarity;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.io.OutputFormat;
import org.apache.flink.api.java.io.CsvOutputFormat;
import org.apache.flink.api.java.tuple.Tuple;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamSink;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.flink.util.Collector;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FlinkRetrievalApp {

  private static final Logger LOG = LoggerFactory.getLogger(FlinkRetrievalApp.class);

  private static ParagraphVectors paragraphVectors;
  private static IndexWriter writer;

  private static void initializeModels() throws IOException {
    paragraphVectors = new ParagraphVectors.Builder()
        .tokenizerFactory(new DefaultTokenizerFactory())
        .trainWordVectors(true)
        .useUnknown(true)
        .iterate(new LabelAwareFileSentenceIterator(new File("src/test/resources/data/text")))
    .build();
  }

  public static void main(String[] args) throws Exception {

    initializeModels();
    Collection<Similarity> similarities = initializeSimilarities();

    Directory directory = FSDirectory.open(Paths.get("targe/index"), NoLockFactory.INSTANCE);
    IndexWriterConfig conf = new IndexWriterConfig();
    writer = new IndexWriter(directory, conf);

    final StreamExecutionEnvironment env =
        StreamExecutionEnvironment.getExecutionEnvironment().setParallelism(1);

    env.getConfig().enableObjectReuse();
    env.setStreamTimeCharacteristic(TimeCharacteristic.IngestionTime);

    // twitter credentials and source
    Properties props = new Properties();
    props.load(FlinkRetrievalApp.class.getResourceAsStream("/twitter.properties"));
    TwitterSource twitterSource = new TwitterSource(props);
    twitterSource.setCustomEndpointInitializer(new TwitterFlinkStreaming.FilterEndpoint("#bbuzz", "#bbuzz18"));

    DataStream<Tweet> twitterStream = env.addSource(twitterSource)
        .filter((FilterFunction<String>) value -> value.contains("created_at"))
        .flatMap(new TweetJsonConverter());

    int batchSize = 1;

    Path path = new Path("src/main/html/data.csv");
    OutputFormat<Tuple> format = new CsvOutputFormat<>(path);
    DataStreamSink<Tuple> tweetSearchStream =
        twitterStream
            .countWindowAll(batchSize)
            .apply(new ModelUpdateFunction(paragraphVectors))
        .map(new IndexFunction(writer))
        .countWindowAll(batchSize)
        .apply(new CommitFunction(writer))
        .map(new SearcherFactoryFunction(similarities))
        .map(new MultiRetrieverFunction())
        .map(new ResultTransformer()).countWindowAll(1)
            .apply(new TupleEvictorFunction())
            .writeUsingOutputFormat(format);

  }

  @NotNull
  private static Collection<Similarity> initializeSimilarities() {
    Collection<Similarity> similarities = new LinkedList<>();
    similarities.add(new BM25Similarity());
    similarities.add(new ClassicSimilarity());
    similarities.add(new LMDirichletSimilarity());
    similarities.add(new ParagraphVectorsSimilarity(paragraphVectors, "text"));
    similarities.add(new WordEmbeddingsSimilarity(paragraphVectors, "text"));
    return similarities;
  }

}
