package de.bbuzz18.nsf;

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.hadoop.shaded.org.codehaus.jackson.JsonNode;
import org.apache.flink.hadoop.shaded.org.codehaus.jackson.map.ObjectMapper;
import org.apache.flink.util.Collector;

public class TweetJsonConverter extends RichFlatMapFunction<String,String> {
  private transient ObjectMapper mapper;

  @Override
  public void open(Configuration parameters) throws Exception {
    super.open(parameters);
    mapper = new ObjectMapper();
  }

  @Override
  public void flatMap(String value, Collector<String> out) {
    String tweetString = null;
    JsonNode tweet = null;

    try {
      tweet = mapper.readValue(value, JsonNode.class);
      tweetString = tweet.get("text").getTextValue();
    } catch (Exception e) {
      // received a malformed document
    }

    if (tweetString != null) {
      out.collect(tweet.get("text").asText());
    }
  }
}