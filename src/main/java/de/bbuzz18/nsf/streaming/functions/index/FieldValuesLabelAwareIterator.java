package de.bbuzz18.nsf.streaming.functions.index;

import java.io.IOException;
import java.util.Collections;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;

/**
 *
 */
public class FieldValuesLabelAwareIterator implements LabelAwareIterator {

  private final IndexReader reader;
  private final String field;
  private final LabelsSource labelSource;
  private int currentId;

  public FieldValuesLabelAwareIterator(IndexReader reader, String field) {
    this.reader = reader;
    this.field = field;
    this.currentId = 0;
    this.labelSource = new LabelsSource();
  }

  @Override
  public boolean hasNextDocument() {
    return currentId < reader.maxDoc();
  }

  @Override
  public LabelledDocument nextDocument() {
    if (!hasNext()) {
      return null;
    }
    try {
      LabelledDocument labelledDocument = new LabelledDocument();
      Document document = reader.document(currentId, Collections.singleton(field));
      String label = "doc_" + currentId;
      labelledDocument.addLabel(label);
      labelledDocument.setId(label);
      labelledDocument.setContent(document.getField(field).stringValue());
      labelSource.storeLabel(label);
      return labelledDocument;
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      currentId++;
    }
  }

  @Override
  public void reset() {
    currentId = 0;
  }

  @Override
  public LabelsSource getLabelsSource() {
    return labelSource;
  }

  @Override
  public void shutdown() {
  }

  @Override
  public boolean hasNext() {
    return hasNextDocument();
  }

  @Override
  public LabelledDocument next() {
    return nextDocument();
  }
}
