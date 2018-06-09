package de.bbuzz18.nsf.streaming;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Properties;

import de.bbuzz18.nsf.streaming.functions.CommitFunction;
import de.bbuzz18.nsf.streaming.functions.IndexFunction;
import de.bbuzz18.nsf.streaming.functions.ModelAndIndexUpdateFunction;
import de.bbuzz18.nsf.streaming.functions.ModelUpdateFunction;
import de.bbuzz18.nsf.streaming.functions.MultiRetrieverFunction;
import de.bbuzz18.nsf.streaming.functions.ResultTransformer;
import de.bbuzz18.nsf.streaming.functions.SearcherFactoryFunction;
import de.bbuzz18.nsf.streaming.functions.TupleEvictorFunction;
import de.bbuzz18.nsf.streaming.functions.TweetJsonConverter;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.io.OutputFormat;
import org.apache.flink.api.java.io.CsvOutputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamSink;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FlinkRetrievalApp {

  private static final Logger LOG = LoggerFactory.getLogger(FlinkRetrievalApp.class);

  public static void main(String[] args) throws Exception {

    DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
        .tokenizerFactory(tokenizerFactory)
        .trainWordVectors(true)
        .useUnknown(true)
        .iterate(new LabelAwareFileSentenceIterator(new File("src/test/resources/data/text/abstracts.txt")))
        .build();
    paragraphVectors.fit();
    WordVectorSerializer.writeParagraphVectors(paragraphVectors, "target/pv.zip");
    paragraphVectors.setTokenizerFactory(tokenizerFactory);

    Directory directory = FSDirectory.open(Paths.get("target/index"), NoLockFactory.INSTANCE);
    IndexWriterConfig conf = new IndexWriterConfig();

    final StreamExecutionEnvironment env =
        StreamExecutionEnvironment.getExecutionEnvironment().setParallelism(1);

    env.getConfig().enableObjectReuse();
    env.setStreamTimeCharacteristic(TimeCharacteristic.IngestionTime);

    Properties props = new Properties();
    props.load(FlinkRetrievalApp.class.getResourceAsStream("/twitter.properties"));
    TwitterSource twitterSource = new TwitterSource(props);
    twitterSource.setCustomEndpointInitializer(new TwitterFlinkStreaming.FilterEndpoint("#bbuzz", "#bbuzz18", "lucene", "berlin", "deep learning", "embeddings", "relevance"));

    DataStream<Tweet> twitterStream = env.addSource(twitterSource)
        .filter((FilterFunction<String>) value -> value.contains("created_at"))
        .flatMap(new TweetJsonConverter());

    int batchSize = 1;

    Path path = new Path("src/main/html/data.csv");
    OutputFormat<Tuple2<String,String>> format = new CsvOutputFormat<>(path);
    DataStreamSink<Tuple2<String,String>> tweetSearchStream =
        twitterStream
            .countWindowAll(batchSize)
            .apply(new ModelAndIndexUpdateFunction(paragraphVectors))
        .map(new SearcherFactoryFunction(paragraphVectors))
        .map(new MultiRetrieverFunction())
        .map(new ResultTransformer()).countWindowAll(1)
            .apply(new TupleEvictorFunction())
            .writeUsingOutputFormat(format);

    env.execute();
  }

}
