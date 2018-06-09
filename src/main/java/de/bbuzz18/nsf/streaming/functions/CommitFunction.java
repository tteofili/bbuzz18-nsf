package de.bbuzz18.nsf.streaming.functions;

import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.util.Collector;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;

/**
 *
 */
public class CommitFunction implements AllWindowFunction<Long, IndexReader, GlobalWindow> {
  private final IndexWriter writer;

  public CommitFunction(IndexWriter writer) {
    this.writer = writer;
  }

  @Override
  public void apply(GlobalWindow globalWindow, Iterable<Long> iterable, Collector<IndexReader> collector) throws Exception {
    writer.commit();
    DirectoryReader reader = DirectoryReader.open(writer);
    System.err.println("****"+reader.numDocs());
    collector.collect(reader);
  }
}
