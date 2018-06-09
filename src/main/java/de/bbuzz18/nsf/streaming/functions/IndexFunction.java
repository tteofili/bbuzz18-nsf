package de.bbuzz18.nsf.streaming.functions;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexWriter;

/**
 *
 */
public class IndexFunction implements MapFunction<Document, Long> {
  private final IndexWriter writer;

  public IndexFunction(IndexWriter writer) {
    this.writer = writer;
  }

  @Override
  public Long map(Document value) throws Exception {
    return writer.addDocument(value);
  }
}
