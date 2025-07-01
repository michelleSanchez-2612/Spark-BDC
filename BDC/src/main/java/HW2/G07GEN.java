package main.java.HW2;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

/*********************+
 * receives in input, as command-line arguments, 2 integers N,K
 * , and generates a dataset of N
 *  points in R^2, which, if used as input for the above program,
 *  show a clear quality gap between the solutions provided by
 *  the standard LLoyd's algorithm and its fair variant.
* */

public class G07GEN {
    public static int N;
    public static int K;


    public static void checkArguments(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: N K");
        }
        try {
            N = Integer.parseInt(args[0]);
            K = Integer.parseInt(args[1]);

        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("USAGE: N K must be integers");
        }
    }

    public static void main(String[] args) throws IOException {
        try {
            checkArguments(args);
            System.out.println("N = " + N + ", K = " + K);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
            System.exit(1);
        }
        Random rand = new Random();
        //rand.setSeed(42);

        // Points of class A: from 10% to 40%
        int percA = 10 + rand.nextInt(31);
        int nA = (N * percA) / 100;
        int nB = N - nA;
        System.out.println("Generating " + nA + " points for group A (" + percA + "%)");
        System.out.println("Generating " + nB + " points for group B (" + (100 - percA) + "%)");

        // Distance between the centers of groups A and B: from 10 to 30
        int distance = 10 + rand.nextInt(21);
        System.out.println("Group B will be centered at distance = " + distance);

        String fileName = "G07GEN_N" + N + "_K" + K + ".txt";

        PrintWriter writer = new PrintWriter(new FileWriter(new File(fileName)));

        //Class A: 1 group of points close to (0,0)
        for (int i = 0; i < nA; i++) {
            double x = rand.nextGaussian() * 0.5;
            double y = rand.nextGaussian() * 0.5;
            //System.out.printf("%.4f %.4f A\n", x, y);
            writer.printf("%.4f,%.4f,A\n", x, y);
        }

        //Group B: K groups of points distant from A and not too close to their center
        int clusters_B = K;
        int pointsPerClusterB = nB / clusters_B;
        int extraPoints = nB % clusters_B; //remainder

        for (int c = 0; c < clusters_B; c++) {
            double centerX = (c + 1) * distance;
            double centerY = (c + 1) * distance;

            // Add the extra points to the last cluster
            int pointsThisCluster = pointsPerClusterB + (c == clusters_B - 1 ? extraPoints : 0);

            for (int i = 0; i < pointsThisCluster; i++) {
                double x = centerX + rand.nextGaussian() * 5.0;
                double y = centerY + rand.nextGaussian() * 5.0;
                //System.out.printf("%.4f %.4f B\n", x, y);
                writer.printf("%.4f,%.4f,B\n", x, y);
            }
        }
        writer.close();

    }
}

