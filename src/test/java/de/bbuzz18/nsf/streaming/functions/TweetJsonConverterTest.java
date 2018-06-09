package de.bbuzz18.nsf.streaming.functions;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.util.LinkedList;

import de.bbuzz18.nsf.streaming.Tweet;
import de.bbuzz18.nsf.streaming.functions.TweetJsonConverter;
import org.apache.commons.io.IOUtils;
import org.apache.flink.api.common.functions.util.ListCollector;
import org.apache.flink.configuration.Configuration;
import org.junit.Test;

import static org.junit.Assert.assertFalse;

/**
 *
 */
public class TweetJsonConverterTest {

  @Test
  public void testConversion() throws Exception {
    LinkedList<Tweet> tweets = new LinkedList<>();
    TweetJsonConverter tweetJsonConverter = new TweetJsonConverter();
    tweetJsonConverter.open(new Configuration());
    ListCollector<Tweet> collector = new ListCollector<>(tweets);
    for (String line : IOUtils.readLines(new FileInputStream("src/test/resources/data/json/bbuzz_mini.json"))) {
      tweetJsonConverter.flatMap(line, collector);
    }
    assertFalse(tweets.isEmpty());
    File f = new File("target/bbuzz_mini.txt");
    OutputStream out = new FileOutputStream(f);
    for (Tweet t : tweets) {
      out.write(t.getText().getBytes(Charset.defaultCharset()));
      out.write("\n".getBytes(Charset.defaultCharset()));
    }
    out.flush();
    out.close();
  }
}