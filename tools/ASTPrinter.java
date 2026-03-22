import com.github.javaparser.*;
import com.github.javaparser.ast.*;

import java.io.File;

public class ASTPrinter {

    public static void main(String[] args) {
        try {
            if (args.length == 0) {
                System.out.println("No file provided");
                return;
            }

            File file = new File(args[0]);

            CompilationUnit cu = StaticJavaParser.parse(file);

            // Print AST nodes
            printNode(cu, 0);

        } catch (Exception e) {
            System.out.println("ERROR: " + e.getMessage());
        }
    }

    private static void printNode(Node node, int level) {
        // Print node type only (clean output)
        System.out.println(node.getClass().getSimpleName());

        for (Node child : node.getChildNodes()) {
            printNode(child, level + 1);
        }
    }
}