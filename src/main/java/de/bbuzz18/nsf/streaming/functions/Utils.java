package de.bbuzz18.nsf.streaming.functions;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import de.bbuzz18.nsf.streaming.functions.index.FieldValuesLabelAwareIterator;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

/**
 *
 */
public class Utils {
  public static ParagraphVectors fetchVectors(java.nio.file.Path indexPath) throws IOException {
    Directory dir = FSDirectory.open(indexPath);
    DirectoryReader reader = DirectoryReader.open(dir);

    DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
        .tokenizerFactory(tokenizerFactory)
        .trainWordVectors(true)
        .layerSize(60)
        .epochs(1)
        .useUnknown(true)
        .iterate(new FieldValuesLabelAwareIterator(reader, "text"))
        .build();
    paragraphVectors.fit();

    reader.close();
    dir.close();
    return paragraphVectors;
  }

  public static ParagraphVectors fetchVectors() throws IOException {
    return fetchVectors(getIndexPath());
  }

  public static Path getIndexPath() {
    return Paths.get("stream_index");
  }
}
