using System.Globalization;

namespace Iris;

public static class IrisDataReader
{
    public static List<Tuple<double[], IrisLabel>> ReadData()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "..","..","..", "sample_training_data");
        var lines = File.ReadAllLines(path);
        
        return lines.Select(ParseSample).ToList();
    }
    
    private static Tuple<double[], IrisLabel> ParseSample(string line)
    {
        var properties = line
            .Split()
            .Select(x => double.Parse(x, CultureInfo.InvariantCulture))
            .ToArray();

        if (properties.Length != 5)
        {
            throw new ArgumentException("Invalid sample line");
        }

        return new Tuple<double[], IrisLabel>(properties.Take(4).ToArray(), (IrisLabel)properties.Last());
    }
}
