package main.java.HW2;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class G07HW2 {
    public static String filePath;
    public static int L;
    public static int K;
    public static int M;

    public static void main(String[] args) throws IOException {
        //Spark Setup
        SparkConf conf = new SparkConf(true).setAppName("G07HW2").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        //Prints the command-line arguments and stores L,K,M, into suitable variables.
        try {
            checkArguments(args);
            System.out.println("Input file = " + filePath + ", L = " + L + ", K = " + K + ", M = " + M);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
            System.exit(1);
        }

        //Reads the input points into an RDD -called inputPoints-, subdivided into L partitions.
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
        JavaPairRDD<String, Long> groupsCountRDD = inputPoints.mapToPair(pair ->
                new Tuple2<>(pair._2(), 1L)
        ).reduceByKey(Long::sum);

        Map<String, Long> groupsCount = groupsCountRDD.collectAsMap();
        System.out.println("N = " + numPoints + ", NA = " + groupsCount.getOrDefault("A", 0L) + ", NB = "+groupsCount.getOrDefault("B", 0L));


        //Computes a set C stand of K centroids with standard Lloyd's algorithm
        long stand_start = System.currentTimeMillis();
        KMeansModel clusters = KMeans.train(inputVectors.rdd(), K, M);
        Vector[] centroids_stand = clusters.clusterCenters();
        long stand_end = System.currentTimeMillis();


        //Computes a set C fair of K centroids by running MRFairLloyd(inputPoints,K,M).
        long fair_start = System.currentTimeMillis();
        Vector[] centroids_fair = MRFairLloyd(inputPoints, K, M);
        long fair_end = System.currentTimeMillis();


        //Computes objective function for standard Lloyd's algorithm
        long obj_stand_start = System.currentTimeMillis();
        double standardObjValue = MRComputeFairObjective(inputPoints, Arrays.asList(centroids_stand));
        System.out.println("Fair Objective with Standard Centers = " + String.format("%.4f", standardObjValue));
        long obj_stand_end = System.currentTimeMillis();


        //Computes objective function for fair Lloyd's algorithm
        long obj_fair_start = System.currentTimeMillis();
        double fairObjValue = MRComputeFairObjective(inputPoints, Arrays.asList(centroids_fair));
        System.out.println("Fair Objective with Fair Centers = " + String.format("%.4f", fairObjValue));
        long obj_fair_end = System.currentTimeMillis();


        //Prints separately the times, in seconds, spent to compute : C stand, C fair, Obj Stand, Obj Fair
        printRunningTimes("Time to compute standard centers = ", stand_end-stand_start);
        printRunningTimes("Time to compute fair centers = ", fair_end-fair_start);
        printRunningTimes("Time to compute objective with standard centers = ", obj_stand_end-obj_stand_start);
        printRunningTimes("Time to compute objective with fair centers = ", obj_fair_end-obj_fair_start);
    }

    public static Vector[] MRFairLloyd(JavaPairRDD<Vector, String> inputPoints, int K, int M) {
        // Initialize centroids using kmeans (Spark implementation of Lloyd's algorithm with 0 iterations)
        JavaRDD<Vector> inputVectors = inputPoints.keys();
        KMeansModel initialModel = KMeans.train(inputVectors.rdd(), K, 0);
        Vector[] centroids = initialModel.clusterCenters();
        int dim = centroids[0].size();

        // Filter the inputPoints RDD to separate points labeled as "A" into groupA and points labeled as "B" into groupB
        JavaRDD<Vector> groupA = inputPoints.filter(p -> p._2().equals("A")).keys().cache();
        JavaRDD<Vector> groupB = inputPoints.filter(p -> p._2().equals("B")).keys().cache();

        // Count the total number of points in each group (A and B)
        long totalA = groupA.count();
        long totalB = groupB.count();

        // Loop M times
        for (int iter = 0; iter < M; iter++) {
            final Vector[] currentCentroids = centroids;

            // Assign each point to its closest centroid (assume that ties are broken in favor of the smallest index)
            JavaPairRDD<Integer, Tuple2<Vector, String>> assigned = inputPoints.mapToPair(p -> {
                Vector point = p._1();
                String label = p._2();

                int closest = 0;
                double minDist = Vectors.sqdist(point, currentCentroids[0]);
                for (int i = 1; i < currentCentroids.length; i++) {
                    double dist = Vectors.sqdist(point, currentCentroids[i]);
                    if (dist < minDist) {
                        closest = i;
                        minDist = dist;
                    }
                }

                return new Tuple2<>(closest, new Tuple2<>(point, label));
            });

            // Compute a new set of k centroids using the CentroidsSelection algorithm we saw in class
            Map<Integer, Iterable<Tuple2<Vector, String>>> clusterData = assigned.groupByKey().collectAsMap();

            // Centroid Selection Algorithm

            // Compute muA, muB, countA, countB
            double[][] muA = new double[K][dim];
            double[][] muB = new double[K][dim];
            int[] countA = new int[K];
            int[] countB = new int[K];

            // 1) First pass: compute sums & counts for each cluster
            for (int i = 0; i < K; i++) {
                Iterable<Tuple2<Vector, String>> clusterPoints = clusterData.get(i);
                if (clusterPoints == null) continue;

                double[] sumA = new double[dim];
                double[] sumB = new double[dim];

                for (Tuple2<Vector, String> p : clusterPoints) {
                    double[] arr = p._1().toArray();
                    String label = p._2();
                    if (label.equals("A")) {
                        countA[i]++;
                        for (int d = 0; d < dim; d++) {
                            sumA[d] += arr[d];
                        }
                    } else {
                        countB[i]++;
                        for (int d = 0; d < dim; d++) {
                            sumB[d] += arr[d];
                        }
                    }
                }

                for (int d = 0; d < dim; d++) {
                    muA[i][d] = countA[i] > 0 ? sumA[d] / countA[i] : 0.0;
                    muB[i][d] = countB[i] > 0 ? sumB[d] / countB[i] : 0.0;
                    muA[i][d] = muA[i][d] == 0.0 ? muB[i][d] : muA[i][d];
                    muB[i][d] = muB[i][d] == 0.0 ? muA[i][d] : muB[i][d];
                }
            }

            // 2) Second pass: compute fixedA and fixedB using the true means
            double fixedA = 0.0;
            double fixedB = 0.0;
            for (int i = 0; i < K; i++) {
                Iterable<Tuple2<Vector, String>> clusterPoints = clusterData.get(i);
                if (clusterPoints == null) continue;

                Vector centerA = Vectors.dense(muA[i]);
                Vector centerB = Vectors.dense(muB[i]);
                for (Tuple2<Vector, String> p : clusterPoints) {
                    if (p._2().equals("A")) {
                        fixedA += Vectors.sqdist(p._1(), centerA);
                    } else {
                        fixedB += Vectors.sqdist(p._1(), centerB);
                    }
                }
            }
            fixedA = totalA > 0 ? fixedA / totalA : 0.0;
            fixedB = totalB > 0 ? fixedB / totalB : 0.0;

            // Compute ℓᵢ = ||μAᵢ − μBᵢ|| for each cluster i
            double[] ell = new double[K];
            for (int i = 0; i < K; i++) {
                double distSq = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = muA[i][d] - muB[i][d];
                    distSq += diff * diff;
                }
                ell[i] = Math.sqrt(distSq);
            }

            // Compute αᵢ = |A ∩ Uᵢ| / |A|,  βᵢ = |B ∩ Uᵢ| / |B|
            double[] alpha = new double[K];
            double[] beta = new double[K];
            for (int i = 0; i < K; i++) {
                alpha[i] = totalA > 0 ? (double) countA[i] / totalA : 0.0;
                beta[i]  = totalB > 0 ? (double) countB[i] / totalB : 0.0;
            }

            // Compute x using fair Lloyd's weights
            double[] x = computeVectorX(fixedA, fixedB, alpha, beta, ell, K);

            // Update centroids
            for (int i = 0; i < K; i++) {
                double[] newCenter = new double[dim];
                double xi = x[i];
                double elli = ell[i];


                for (int d = 0; d < dim; d++) {
                    if (elli == 0) {
                        newCenter[d] = muA[i][d]; // or muB[i][d], they are the same
                    } else{
                        newCenter[d] = ((elli - xi) * muA[i][d] + xi * muB[i][d]) / elli;
                    }
                }

                centroids[i] = Vectors.dense(newCenter);
            }
        }

        return centroids;
    }


    //Compute vectorX for selection algorithm
    public static double[] computeVectorX(double fixedA, double fixedB, double[] alpha, double[] beta, double[] ell, int K) {
        double gamma = 0.5;
        double[] xDist = new double[K];
        double fA, fB;
        double power = 0.5;
        int T = 10;
        for (int t=1; t<=T; t++){
            fA = fixedA;
            fB = fixedB;
            power = power/2;
            for (int i=0; i<K; i++) {
                double temp = (1-gamma)*beta[i]*ell[i]/(gamma*alpha[i]+(1-gamma)*beta[i]);
                xDist[i]=temp;
                fA += alpha[i]*temp*temp;
                temp=(ell[i]-temp);
                fB += beta[i]*temp*temp;
            }
            if (fA == fB) {break;}
            gamma = (fA > fB) ? gamma+power : gamma-power;
        }
        return xDist;
    }

    //Check command line arguments
    public static void checkArguments(String[] args){
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

    //Compute objective fair function
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

    //Print running times
    public static void printRunningTimes(String name, long dif) {
        System.out.println(name + dif + " ms");
    }
}
