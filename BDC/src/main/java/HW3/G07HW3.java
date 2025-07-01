package main.java.HW3;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.util.*;
import java.util.concurrent.Semaphore;

public class G07HW3 {
    // Streaming and sketch parameters
    public static int portExp, T, D, W, K;
    public static final Set<Integer> legalPorts = new HashSet<>(Arrays.asList(8886, 8887, 8888, 8889));
    public static final String streamingServer = "algo.dei.unipd.it";
    public static final int P = 8191;

    public static void main(String[] args) throws Exception {
        // === Spark Setup ===
        SparkConf conf = new SparkConf(true).setAppName("G07HW3").setMaster("local[*]");
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        // === Parse & Validate Arguments ===
        try {
            checkArguments(args);
            System.out.printf("Port = %d T = %d D = %d W = %d K = %d%n", portExp, T, D, W, K);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
            System.exit(1);
        }

        // === Initialize Hash Functions ===
        Random rand = new Random();
        int[] a_h_cm = new int[D], b_h_cm = new int[D]; // for CM Sketch
        int[] a_h_cs = new int[D], b_h_cs = new int[D]; // for Count Sketch
        int[] a_g_cs = new int[D], b_g_cs = new int[D]; // for Count Sketch sign

        for (int i = 0; i < D; i++) {
            a_h_cm[i] = 1 + rand.nextInt(P - 1);
            b_h_cm[i] = rand.nextInt(P);

            a_h_cs[i] = 1 + rand.nextInt(P - 1);
            b_h_cs[i] = rand.nextInt(P);

            a_g_cs[i] = 1 + rand.nextInt(P - 1);
            b_g_cs[i] = rand.nextInt(P);
        }

        // === Structures for Tracking State ===
        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire(); // will release when done

        long[] streamLength = new long[] {0L};
        HashMap<Long, Long> exactFrequencies = new HashMap<>();

        long[][] cmSketch = new long[D][W];
        long[][] countSketch = new long[D][W];

        // === Stream Processing ===
        sc.socketTextStream(streamingServer, portExp, StorageLevels.MEMORY_AND_DISK)
                .foreachRDD((batch, time) -> {
                    if (streamLength[0] < T) {
                        long batchSize = batch.count();
                        streamLength[0] += batchSize;

                        if (batchSize > 0) {
                            //System.out.println("Batch size at time [" + time + "] is: " + batchSize);

                            Map<Long, Long> batchItems = batch
                                    .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L))
                                    .reduceByKey(Long::sum)
                                    .collectAsMap();

                            for (Map.Entry<Long, Long> entry : batchItems.entrySet()) {
                                long item = entry.getKey();
                                long freq = entry.getValue();

                                // Exact histogram
                                exactFrequencies.merge(item, freq, Long::sum);

                                // Count-Min Sketch update
                                updateCMS(cmSketch, a_h_cm, b_h_cm, W, item, freq);

                                // Count Sketch update
                                updateCountSketch(countSketch, a_h_cs, b_h_cs, a_g_cs, b_g_cs, W, item, freq);
                            }

                            if (streamLength[0] >= T) {
                                stoppingSemaphore.release();
                            }
                        }
                    }
                });

        // === Start & Wait for Shutdown ===
        sc.start();
        stoppingSemaphore.acquire();
        sc.stop(false, false);

        // === Final Report ===
        System.out.println("Number of processed items = " + streamLength[0]);
        System.out.println("Number of distinct items  = " + exactFrequencies.size());

        Map<Long, Long> estCS = new HashMap<>();
        Map<Long, Long> estCM = new HashMap<>();

        for (Map.Entry<Long, Long> entry : exactFrequencies.entrySet()) {
            long item = entry.getKey();
            long real = entry.getValue();
            if (real > 1) {
                long estCMValue = estimateCMS(cmSketch, a_h_cm, b_h_cm, W, item);
                long estCSValue = estimateCS(countSketch, W, a_h_cs, b_h_cs, a_g_cs, b_g_cs, item);
                estCM.put(item, estCMValue);
                estCS.put(item, estCSValue);
            }
        }

        //Calculate average error
        calculateTopKRelativeError(exactFrequencies, estCM, estCS);
    }

    // === Utilities ===
    public static int computeHashFunction(int a, int b, int C, long x) {
        return (int)(((a * x + b) % P) % C);
    }

    // === Count Sketch Utilities ===
    public static int computeSignFunctionCS(int a, int b, int C, long x) {
        int hash = (int)(((a * x + b) % P) % C);
        return (hash % 2 == 0) ? 1 : -1;
    }

    public static void updateCountSketch(long[][] countSketch, int[] a_h_cs, int[] b_h_cs, int[] a_g_cs, int[] b_g_cs, int W, long item, long freq) {
        for (int d = 0; d < D; d++) {
            int hash = computeHashFunction(a_h_cs[d], b_h_cs[d], W, item);
            int sign = computeSignFunctionCS(a_g_cs[d], b_g_cs[d], 2, item);
            countSketch[d][hash] += sign * freq;
        }
    }

    public static long estimateCS(long[][] sketch, int W, int[] a_h, int[] b_h, int[] a_g, int[] b_g, long x) {
        long[] estimates = new long[D];
        for (int d = 0; d < D; d++) {
            int hash = computeHashFunction(a_h[d], b_h[d], W, x);
            int sign = computeSignFunctionCS(a_g[d], b_g[d], 2, x);
            estimates[d] = sign * sketch[d][hash];
        }
        Arrays.sort(estimates);
        return (D % 2 == 0) ? (estimates[D / 2] + estimates[D / 2 - 1]) / 2 : estimates[D / 2];
    }

    // === Count-Min Sketch Utilities ===
    public static void updateCMS(long[][] cmSketch, int[] a_h_cs, int[] b_h_cs, int W, long item, long freq) {
        for (int d = 0; d < D; d++) {
            int hash = computeHashFunction(a_h_cs[d], b_h_cs[d], W, item);
            cmSketch[d][hash] += freq;
        }
    }


    public static long estimateCMS(long[][] cmSketch, int[] a_h, int[] b_h, int W, long el) {
        long estimate = Long.MAX_VALUE;
        for (int d = 0; d < D; d++) {
            int hash = computeHashFunction(a_h[d], b_h[d], W, el);
            estimate = Math.min(estimate, cmSketch[d][hash]);
        }
        return estimate;
    }

    // === Error Function ===
    public static void calculateTopKRelativeError(Map<Long, Long> exactFrequencies,Map<Long, Long> estCM, Map<Long, Long> estCS) {
        // Sort frequencies
        List<Long> sortedFrequencies = new ArrayList<>(exactFrequencies.values());
        sortedFrequencies.sort(Comparator.reverseOrder());

        if (sortedFrequencies.size() < K) {
            System.out.println("There are not enough elements to calculate top-K");
            return;
        }

        //Get phi value
        long phiK = sortedFrequencies.get(K - 1);

        //Frequent values >= ϕ(K)
        List<Long> topKItems = new ArrayList<>();
        for (Map.Entry<Long, Long> entry : exactFrequencies.entrySet()) {
            if (entry.getValue() >= phiK) {
                topKItems.add(entry.getKey());
            }
        }

        // Calculate relative error for the topK items
        double sumErrorCS = 0.0;
        double sumErrorCM = 0.0;

        for (Long item : topKItems) {
            long real = exactFrequencies.get(item);
            long estimateCM = estCM.getOrDefault(item, 0L);
            long estimateCS = estCS.getOrDefault(item, 0L);

            double errorCM = Math.abs(real - estimateCM) / (double) real;
            double errorCS = Math.abs(real - estimateCS) / (double) real;

            sumErrorCS += errorCS;
            sumErrorCM += errorCM;
        }

        double avgErrorCM = sumErrorCM / topKItems.size();
        double avgErrorCS = sumErrorCS / topKItems.size();


        System.out.println("Number of Top-K Heavy Hitters = " + topKItems.size());
        System.out.printf("Avg Relative Error for Top-K Heavy Hitters with CM = %.14f\n", avgErrorCM);
        System.out.printf("Avg Relative Error for Top-K Heavy Hitters with CS = %.14f\n", avgErrorCS);

        //(Only if K≤10) True and estimated frequencies of the top-K heavy hitters. For the estimated frequencies consider only those of CM.
        if (K <= 10) {
            System.out.println("\nTop-K Heavy Hitters:");
            topKItems.sort(Long::compare);

            int count = 0;
            for (Long item : topKItems) {
                if (count >= K) break;
                long real = exactFrequencies.get(item);
                long estimateCM = estCM.getOrDefault(item, 0L);
                System.out.printf("Item %d True Frequency = %d Estimated Frequency with CM = %d%n", item, real, estimateCM);
                count++;
            }
        }
    }


    // === Argument Validation ===
    public static void checkArguments(String[] args) {
        if (args.length != 5)
            throw new IllegalArgumentException("USAGE: port T D W K");
        try {
            portExp = Integer.parseInt(args[0]);
            T = Integer.parseInt(args[1]);
            D = Integer.parseInt(args[2]);
            W = Integer.parseInt(args[3]);
            K = Integer.parseInt(args[4]);

            if (!legalPorts.contains(portExp))
                throw new IllegalArgumentException("Invalid port. Allowed: " + legalPorts);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Arguments must be integers: portExp T D W K");
        }
    }
}
