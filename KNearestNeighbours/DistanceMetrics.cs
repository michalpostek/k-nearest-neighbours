namespace KNearestNeighbours;

public static class DistanceMetrics
{
    public static double ManhattanDistance(double[] from, double[] to)
    {
        ThrowIfInvalidArgs(from, to);

        return from.Select((x, index) => Math.Abs(x - to[index])).Sum();
    }

    public static double EuclideanDistance(double[] from, double[] to)
    {
        ThrowIfInvalidArgs(from, to);

        var sum = from.Select((x, index) => Math.Pow(x - to[index], 2)).Sum();

        return Math.Sqrt(sum);
    }

    public static double ChebyshevDistance(double[] from, double[] to)
    {
        ThrowIfInvalidArgs(from, to);

        var diffs = from.Select((x, index) => Math.Abs(x - to[index]));

        return diffs.Max();
    }

    public static double MinkowskiDistance(double[] from, double[] to, int p)
    {
        ThrowIfInvalidArgs(from, to);
        
        if (p < 1)
        {
            throw new ArgumentException("p must be greater than 0");
        }
        
        var sum = from.Select((x, index) => Math.Pow(Math.Abs(x - to[index]), p)).Sum();

        return Math.Pow(sum, 1.0 / p);
    }

    public static double LogarithmicDistance(double[] from, double[] to)
    {
        ThrowIfInvalidArgs(from, to);

        return from.Select((x, index) => Math.Abs(Math.Log10(x) - Math.Log10(to[index]))).Sum();
    }

    private static void ThrowIfInvalidArgs(double[] from, double[] to)
    {
        if (from.Length != to.Length)
        {
            throw new ArgumentException("from and to must have the same length");
        }
    }
}
