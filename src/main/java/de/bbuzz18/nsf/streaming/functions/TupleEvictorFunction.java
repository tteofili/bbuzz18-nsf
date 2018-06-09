package de.bbuzz18.nsf.streaming.functions;

import java.util.Collection;

import org.apache.flink.api.java.tuple.Tuple;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.util.Collector;

/**
 *
 */
public class TupleEvictorFunction implements AllWindowFunction<Collection<Tuple>, Tuple, GlobalWindow> {

  @Override
  public void apply(GlobalWindow globalWindow, Iterable<Collection<Tuple>> iterable, Collector<Tuple> collector) throws Exception {
    for (Collection<Tuple> c : iterable) {
      for (Tuple t : c) {
        collector.collect(t);
      }
    }
  }
}
