package de.bbuzz18.nsf;

import java.util.Properties;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FlinkRetrievalApp {

  private static final Logger LOG = LoggerFactory.getLogger(FlinkRetrievalApp.class);


  public static void main(String[] args) throws Exception {

    ParameterTool parameterTool = ParameterTool.fromArgs(args);

    final StreamExecutionEnvironment env =
        StreamExecutionEnvironment.getExecutionEnvironment();

    env.getConfig().enableObjectReuse();
    env.setStreamTimeCharacteristic(TimeCharacteristic.IngestionTime);

    // twitter credentials and source
    Properties props = new Properties();
    props.load(FlinkRetrievalApp.class.getResourceAsStream("/twitter.properties"));
    TwitterSource twitterSource = new TwitterSource(props);
//    twitterSource.setCustomEndpointInitializer(
//        new StreamingNmt.FilterEndpoint("#ShakespeareSunday, #SundayMotivation"));

    // Create a DataStream from TwitterSource filtered by deleted tweets
    // filter for en tweets
    DataStream<Tweet> twitterStream = env.addSource(twitterSource)
        .filter((FilterFunction<String>) value -> value.contains("created_at"))
        .flatMap(new TweetJsonConverter());
  }
}
