import java.util.*;
public class BFSExample {
    private Map<Integer, List<Integer>> adjList;
    public BFSExample() {
        adjList = new HashMap<>();
    }
    public void addEdge(int u, int v) {
        adjList.putIfAbsent(u, new ArrayList<>());
        adjList.get(u).add(v);
    }
    public void bfs(int start) {
        // Track visited nodes
        Set<Integer> visited = new HashSet<>();
        // Use a queue to explore nodes level by level
        Queue<Integer> queue = new LinkedList<>();
        visited.add(start);
        queue.add(start);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            System.out.print(node + " ");
            for (int neighbor : adjList.getOrDefault(node, new ArrayList<>())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.add(neighbor);
                }
            }
        }
    }
    public static void main(String[] args) {
        BFSExample graph = new BFSExample();
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 2);
        graph.addEdge(2, 0);
        graph.addEdge(2, 3);
        graph.addEdge(3, 3);
        System.out.println("Breadth First Traversal starting from node 2:");
        graph.bfs(2);
    }
}
