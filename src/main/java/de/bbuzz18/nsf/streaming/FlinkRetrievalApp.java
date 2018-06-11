package de.bbuzz18.nsf.streaming;

import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import com.twitter.hbc.core.endpoint.StatusesFilterEndpoint;
import com.twitter.hbc.core.endpoint.StreamingEndpoint;
import de.bbuzz18.nsf.streaming.functions.ModelAndIndexUpdateFunction;
import de.bbuzz18.nsf.streaming.functions.MultiRetrieverFunction;
import de.bbuzz18.nsf.streaming.functions.ResultTransformer;
import de.bbuzz18.nsf.streaming.functions.TupleEvictorFunction;
import de.bbuzz18.nsf.streaming.functions.TweetJsonConverter;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.io.OutputFormat;
import org.apache.flink.api.java.io.CsvOutputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.Path;
import org.apache.flink.shaded.guava18.com.google.common.collect.Lists;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamSink;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Serializable;

public class FlinkRetrievalApp {

  private static final Logger log = LoggerFactory.getLogger(FlinkRetrievalApp.class);

  public static void main(String[] args) throws Exception {

    final StreamExecutionEnvironment env =
        StreamExecutionEnvironment.getExecutionEnvironment().setParallelism(1);

    env.getConfig().enableObjectReuse();
    env.setStreamTimeCharacteristic(TimeCharacteristic.IngestionTime);

    Properties props = new Properties();
    props.load(FlinkRetrievalApp.class.getResourceAsStream("/twitter.properties"));
    TwitterSource twitterSource = new TwitterSource(props);
    String[] tags = {"#bbuzz", "#bbuzz18", "lucene", "deep learning", "embeddings", "relevance"};
    twitterSource.setCustomEndpointInitializer(new FilterEndpoint(tags));
    log.info("ingesting tweets for tags {}", Arrays.toString(tags));

    DataStream<Tweet> twitterStream = env.addSource(twitterSource)
        .filter((FilterFunction<String>) value -> value.contains("created_at"))
        .flatMap(new TweetJsonConverter());

    int batchSize = 5;

    Path outputPath = new Path("src/main/html/data.csv");
    OutputFormat<Tuple2<String, String>> format = new CsvOutputFormat<>(outputPath);
    DataStreamSink<Tuple2<String, String>> tweetSearchStream =
        twitterStream
            .countWindowAll(batchSize)
            .apply(new ModelAndIndexUpdateFunction())
            .map(new MultiRetrieverFunction())
            .map(new ResultTransformer()).countWindowAll(1)
            .apply(new TupleEvictorFunction())
            .writeUsingOutputFormat(format);

    env.execute();

  }

  static class FilterEndpoint implements TwitterSource.EndpointInitializer, Serializable {

    private final List<String> tags;

    FilterEndpoint(final String... tags) {
      this.tags = Lists.newArrayList(tags);
    }

    @Override
    public StreamingEndpoint createEndpoint() {
      StatusesFilterEndpoint ep = new StatusesFilterEndpoint();
      ep.trackTerms(tags);
      return ep;
    }
  }


}
