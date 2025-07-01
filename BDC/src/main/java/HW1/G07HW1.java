package main.java.HW1;
import java.util.*;
import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;

public class G07HW1 {
    public  static String filePath;
    public static int L;
    public static int K;
    public static int M;

   public static void main(String[] args) throws IOException {
       // SPARK SETUP
       SparkConf conf = new SparkConf(true).setAppName("G07HW1").setMaster("local[*]");
       JavaSparkContext sc = new JavaSparkContext(conf);
       sc.setLogLevel("WARN");

       //Prints the command-line arguments and stores L,K,M, into suitable variables.
       try{
           checkArguments(args);
           System.out.println("Input file = " + filePath +", L = " + L + ", K = " + K + ", M = " + M);
       } catch (IllegalArgumentException e) {
           System.out.println(e.getMessage());
           System.exit(1);
       }

       //Reads the input points into an RDD of (point,group) pairs -called inputPoints-, subdivided into L partitions.
       JavaPairRDD<Vector, String> inputPoints;
       JavaRDD<String> file = sc.textFile(filePath).repartition(L).cache();
       inputPoints = file.mapToPair((point) -> {
           String[] split_string = point.split(",");
           double[] coordinates = new double[split_string.length - 1];

           for (int i = 0; i < coordinates.length; i++) {
               coordinates[i] = Double.parseDouble(split_string[i]);
           }
           Vector v = Vectors.dense(coordinates);
           return new Tuple2<>(v, split_string[split_string.length - 1]);
       });
       long numPoints = inputPoints.count();
       JavaRDD<Vector> inputVectors = inputPoints.keys();

       //Prints the number N of points, the number NA of points of group A, and the number NB of points of group B
       //to fix with the correct output
       JavaPairRDD<String, Long> groupsCountRDD = inputPoints.mapToPair(pair ->
               new Tuple2<>(pair._2(), 1L)
       ).reduceByKey(Long::sum);

       Map<String, Long> groupsCount = groupsCountRDD.collectAsMap();

       //System.out.println("----------POINTS INFORMATION----------");
       System.out.println("N = " + numPoints + ", NA = " + groupsCount.getOrDefault("A", 0L) + ", NB = "+groupsCount.getOrDefault("B", 0L));

       //Computes a set C of K centroids by using the Spark implementation of the standard Lloyd's algorithm
       //inputPoints.keys() to get only the coordinates
       KMeansModel clusters = KMeans.train(inputVectors.rdd(), K, M);
       Vector[] centroids = clusters.clusterCenters();


       //Prints the values of the two objective functions
       double standardObjValue = MRComputeStandardObjective(inputVectors, Arrays.asList(centroids));
       System.out.println("Delta(U,C) = " + String.format("%.6f", standardObjValue));

       double fairObjValue = MRComputeFairObjective(inputPoints, Arrays.asList(centroids));
       System.out.println("Phi(A,B,C) = " + String.format("%.6f", fairObjValue));

       //Runs MRPrintStatistics
       MRPrintStatistics(inputPoints, centroids);

   }

    //Check command line arguments
    public static void checkArguments(String[] args) throws IOException {
       if (args.length != 4) {
           throw new IllegalArgumentException("USAGE: file_path L K M");
       }
       try {
           filePath = args[0];
           L = Integer.parseInt(args[1]);
           K = Integer.parseInt(args[2]);
           M = Integer.parseInt(args[3]);


       } catch (NumberFormatException e) {
           throw new IllegalArgumentException("USAGE: L, K e M must be integers");
       }
    }

    public static double MRComputeStandardObjective(JavaRDD<Vector> points, List<Vector> centroids) {
        long totalPoints = points.count();

        if (totalPoints == 0 || centroids.isEmpty()) {
            return 0.0;
        }

        //Sum of Euclidean distance
        double sumOfSquaredDistances = points.mapToDouble(point -> {
            double minDist = Double.POSITIVE_INFINITY;
            for (Vector c : centroids) {
                double dist = Vectors.sqdist(point, c);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            return minDist;
        }).sum();

        return sumOfSquaredDistances / totalPoints;
    }

    public static double MRComputeFairObjective(JavaPairRDD<Vector, String> inputPoints, List<Vector> centroids) {
        //Divide groups in A and B
        JavaRDD<Vector> groupA = inputPoints
                .filter(t -> t._2().equals("A"))
                .keys();

        JavaRDD<Vector> groupB = inputPoints
                .filter(t -> t._2().equals("B"))
                .keys();

        //Calculate distances (is the same Euclidean distance)
        double deltaA = MRComputeStandardObjective(groupA, centroids);
        double deltaB = MRComputeStandardObjective(groupB, centroids);

        //Get the maximum value
        return Math.max(deltaA, deltaB);
    }

    public static void MRPrintStatistics(JavaPairRDD<Vector, String> inputPoints, Vector[] centroids){
        //MAP REDUCE ALGORITHM
        //Input: pairs (coordinates, group)
        //Map: assign each point to a centroid and transform tuples of group A to (1,0) and group B to (0, 1)
        //Reduce: count the values of group A and B

        JavaPairRDD<Integer, Tuple2<Integer, Integer>> clusterCounts = inputPoints
                .mapToPair(tuple -> {
                    //Assign each point to a centroid
                    Vector point = tuple._1();
                    String group = tuple._2(); //A or B
                    int bestCentroid = 0;
                    double minDist = Double.POSITIVE_INFINITY;
                    for (int i = 0; i < centroids.length; i++) {
                        double dist = Vectors.sqdist(point, centroids[i]);
                        if (dist < minDist) {
                            minDist = dist;
                            bestCentroid = i;
                        }
                    }
                    return new Tuple2<>(bestCentroid, group);
                }).mapValues(group -> {
                    //Transforms A in (1,0) and B(0, 1)
                    if (group.equals("A")) {
                        return new Tuple2<>(1, 0); // (countA, countB)
                    } else {
                        return new Tuple2<>(0, 1);
                    }
                })
                .reduceByKey((p1, p2) -> {
                    //add counts for each cluster
                    int sumA = p1._1() + p2._1();
                    int sumB = p1._2() + p2._2();
                    return new Tuple2<>(sumA, sumB);
                });
        List<Tuple2<Integer, Tuple2<Integer, Integer>>> results = clusterCounts.collect();

        //Transform results to hashmap
        Map<Integer, Tuple2<Integer, Integer>> clusterMap = new HashMap<>();
        for (Tuple2<Integer, Tuple2<Integer, Integer>> entry : results) {
            clusterMap.put(entry._1(), entry._2());
        }

        //Print information
        for (int i = 0; i < centroids.length; i++) {
            Tuple2<Integer, Integer> counts = clusterMap.getOrDefault(i, new Tuple2<>(0, 0));

            int countA = counts._1();
            int countB = counts._2();
            String c_coordinates = "(";
            for (int j = 0; j < centroids[i].size(); j++) {
                c_coordinates += String.format("%.6f", centroids[i].apply(j));
                if (j < centroids[i].size() - 1) {
                    c_coordinates += ",";
                }
            }
            c_coordinates += ")";

            System.out.println("i = " + i + ", center = " + c_coordinates
                    + ", NA" + i + " = " + countA
                    + ", NB" + i + " = " + countB);
        }
    }

    private static String vectorToString(Vector v) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        double[] arr = v.toArray();
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
