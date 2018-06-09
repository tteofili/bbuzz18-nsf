package de.bbuzz18.nsf.streaming;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;

public class CustomWriter extends IndexWriter implements Serializable {
  private Directory directory;
  private IndexWriterConfig iwConfig;

  public CustomWriter() throws IOException {
    this(FSDirectory.open(Paths.get("target/index"), NoLockFactory.INSTANCE), new IndexWriterConfig());
  }

  CustomWriter(Directory dir, IndexWriterConfig iwConf) throws IOException {
    super(dir, iwConf);
  }

  public Directory getDirectory() {
    return directory;
  }

  public IndexWriterConfig getConfig() {
    return iwConfig;
  }

  public void setDirectory(Directory dir) {
    directory = dir;
  }

  public void setConfig(IndexWriterConfig conf) {
    iwConfig = conf;
  }

  public Path getPath() {
    return Paths.get("target/index");
  }
}