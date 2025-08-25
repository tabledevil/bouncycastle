# bouncycastle# BouncyCastle

**A beautiful website viewer for XML files with a focus on topology, schema detection, and smart viewing functions for large and convoluted XML files, especially those with the Genesis schema.**

BouncyCastle is a Node.js project that provides a web-based interface for viewing and analyzing XML files. It's designed to be visually appealing and highly interactive, making it easy to understand the structure and content of complex XML documents. With a particular emphasis on the Genesis schema, BouncyCastle offers intelligent features to handle large and deeply nested XML files, providing a clear topological overview and intuitive exploration tools.

## Features

BouncyCastle comes packed with features designed to provide the best experience for viewing and understanding your XML files.

### 💎 Beautiful & Intuitive Interface
*   **Modern Web UI:** A clean, responsive, and aesthetically pleasing interface that makes navigating XML data a delight.
*   **Light & Dark Modes:** Switch between light and dark themes for comfortable viewing in any environment.
*   **Customizable Themes:** Easily customize the color scheme and layout to your preference.

### 🌳 Advanced XML Tree Explorer
*   **Hierarchical Tree View:** Displays the XML document as a nested tree, allowing you to expand and collapse nodes to explore the hierarchy.
*   **Syntax Highlighting:** Color-coded tags, attributes, and values for improved readability.
*   **Code Folding:** Collapse and expand large sections of the XML to focus on specific areas.
*   **Search & Filter:** Instantly find nodes, attributes, or values within the XML tree.

### 🗺️ Topology & Schema Detection
*   **Automated Schema Inference:** BouncyCastle intelligently analyzes the loaded XML to infer its structure and recurring patterns, which is especially useful when no formal schema is defined, as is common with Genesis XML files.
*   **Visual Topology Graph:** Generates a dynamic and interactive graph that visualizes the relationships between different nodes in the XML. This is perfect for understanding the high-level structure of your data.
*   **Schema Validation (on-the-fly):** If an XSD schema is available, BouncyCastle can validate the XML file against it and highlight any errors or warnings.

### 🔬 Smart Viewing Functions for Large & Convoluted Files
*   **Lazy Loading:** For exceptionally large XML files, BouncyCastle only renders the parts of the tree that are currently in view, ensuring a smooth and responsive experience.
*   **Data Summarization:** Collapses repetitive nodes into a summary view, showing the count of collapsed items. This is ideal for list-like data in Genesis files.
*   **Intelligent Content Preview:** For nodes with large amounts of text content, a truncated preview is shown with an option to expand to the full view.
*   **Attribute Filtering:** Choose to show or hide specific attributes across the entire document to reduce visual clutter.

### ✨ Genesis Schema Specific Enhancements
*   **Genesis Data Recognition:** Automatically identifies common Genesis XML structures, such as character sheets, item lists, and rule sets.
*   **Pre-defined Views:** Offers specialized views for recognized Genesis data types to present the information in a more human-readable format (e.g., displaying a character's stats in a formatted table).
*   **Cross-file Referencing:** If you load multiple related Genesis XML files, BouncyCastle can identify and create hyperlinks between references (e.g., an item in a character's inventory linking to its definition in an items file).

### ↔️ Multiple Viewing Modes
*   **Raw XML View:** A traditional, text-based view of the XML file with syntax highlighting.
*   **Tree View:** The interactive hierarchical tree explorer.
*   **Topology View:** The graphical representation of the XML's structure.
*   **Genesis View:** The specialized, human-readable view for recognized Genesis schema files.

## How to Use It

Getting started with BouncyCastle is simple. Follow these steps to get it up and running on your local machine.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/bouncycastle.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd bouncycastle
    ```
3.  **Install the dependencies:**
    ```bash
    npm install
    ```

### Running the Application

1.  **Start the server:**
    ```bash
    npm start
    ```
2.  **Open your browser:**
    Navigate to `http://localhost:3000` to access the BouncyCastle web interface.

### Viewing an XML File

1.  **Drag and Drop:** The easiest way to view an XML file is to simply drag it from your file explorer and drop it onto the designated area in the BouncyCastle web page.
2.  **File Picker:** Alternatively, you can click the "Upload XML" button to open a file dialog and select the XML file you wish to view.

Once your XML file is loaded, you can use the various features and viewing modes to explore its content and structure.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue on the GitHub repository. If you'd like to contribute code, please fork the repository and submit a pull request.