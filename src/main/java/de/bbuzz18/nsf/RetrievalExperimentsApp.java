package de.bbuzz18.nsf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;

import lucene4ir.ExampleStatsApp;
import lucene4ir.IndexerApp;
import lucene4ir.RetrievalApp;
import org.apache.commons.io.IOUtils;
import org.jetbrains.annotations.NotNull;

/**
 * Sample evaluation program based on Lucene4IR
 */
public class RetrievalExperimentsApp {

  private static String[] measures = new String[] {"ndcg", "map", "Rprec", "recip_rank"};

  public static void main(String[] args) throws Exception {

    File file = new File("output.csv");
    if (!file.exists()) {
      file.createNewFile();
    }
    StringBuilder builder = new StringBuilder();
    builder.append("retrieval,");

    for (String measure : measures) {
      builder.append(measure).append(',');
    }

    if (args != null && args.length > 0) {
      for (String directory : args) {
        FileFilter filter = pathname -> !pathname.getName().startsWith(".");
        File paramsFileDirectory = new File(directory);
        File[] retrievalParamFiles = paramsFileDirectory.listFiles(filter);

        if (retrievalParamFiles != null) {
          builder.append("\n");
          IndexerApp.main(new String[] {"src/test/resources/params/index/simple.xml"});
          for (File retrievalParamFile : retrievalParamFiles) {
            RetrievalApp.main(new String[] {retrievalParamFile.getAbsolutePath()});
            ExampleStatsApp.main(new String[] {"src/test/resources/params/stats/simple.xml"});

            String name = retrievalParamFile.getName();
            builder.append(name).append(',');

            StringBuilder metrics = new StringBuilder();
            for (String measure : measures) {
              String output = trecEval(name, measure);
              metrics.append(output).append(',');
            }

            String line = metrics.toString();

            builder.append(line).append('\n');
          }
        }
      }
    }

    IOUtils.write(builder.toString(), new FileOutputStream(file));
  }

  @NotNull
  private static String trecEval(String name, String measure) {
    Command obj = new Command();

    String command = "/Users/teofili/programs/trec_eval.9.0/trec_eval /Users/teofili/dev/bbuzz18-nsf/data/cacm/cacm.qrels /Users/teofili/dev/bbuzz18-nsf/data/cacm/" + name.substring(0, name.length() - 4) + ".res -m " + measure;

    String output = obj.execute(command);

    int beginIndex = output.indexOf("all") + 4;
    int endIndex = beginIndex + 6;
    output = output.substring(beginIndex, endIndex);
    return output;
  }

  private static class Command {

    private String execute(String command) {

      StringBuilder output = new StringBuilder();

      Process p;
      try {
        p = Runtime.getRuntime().exec(command);
        p.waitFor();
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));

        String line;
        while ((line = reader.readLine()) != null) {
          output.append(line).append('\n');
        }

      } catch (Exception e) {
        e.printStackTrace();
      }

      return output.toString();

    }
  }
}