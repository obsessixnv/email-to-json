<h1>Email Signature Extraction Project</h1>

<h2>Overview</h2>

This project aims to extract and structure relevant information from email signatures into a JSON format using two different models: llama3-8b-8192 and mixtral-8x7b-32768. The extracted information includes fields such as Name, Title, Company, Email, Phone, Address, and Social Media Links.

<h2>Project Structure</h2>

<pre>
<code>
email-signature-extraction/
│
├── api.py                 # Contains the API key for accessing the Groq client
├── main.py                # Main script for running the email signature extraction
├── test_cases.py          # Test cases with various email signatures
├── output_llama3.json     # JSON output generated by the llama3 model
├── output_mixtral.json    # JSON output generated by the mixtral model
└── README.md              # Project overview and instructions
</code>
</pre>


<h2>Prerequisites</h2>
<ul>
  <li>Python 3.8+</li>
  <li><code>groq</code> library (install via pip)</li>
  <li>API key for accessing the Groq client</li>
</ul>


<h2>Usage</h2>
<ol>
  <li>Modify <code>test_cases.py</code> to include the email signatures you want to test.</li>
  <li>Run the main script:
    <pre><code>python main.py</code></pre>
  </li>
  <li>The script will generate JSON files (<code>output_llama3.json</code> and <code>output_mixtral.json</code>) containing the extracted information from the email signatures.</li>
</ol>
<h2>Comparison Summary</h2>

<h3>Core Fields</h3>
<ul>
  <li><strong>Name:</strong> Both models correctly extract the name.</li>
  <li><strong>Position:</strong>  Both are accurate.</li>
  <li><strong>Company:</strong> Both models accurately extract the company name.</li>
  <li><strong>Email:</strong> Both models correctly extract the email address.</li>
  <li><strong>Address:</strong> Both leave the address field empty if not provided.</li>
</ul>

<h3>Phone Number Extraction</h3>
<ul>
  <li><strong>Llama:</strong> Merges phone numbers into a single field (e.g., "Cell: +1928374650").</li>
  <li><strong>Mixtral:</strong> Provides structured details with separate fields for office and cell numbers (e.g., { "office": "+9876543210", "cell": "+1928374650" }).</li>
</ul>

<h3>Social Media Information</h3>
<ul>
  <li><strong>Llama:</strong> Includes an empty or unclear social media field (e.g., { "": "" }).</li>
  <li><strong>Mixtral:</strong> Provides structured placeholders for social media (e.g., { "LinkedIn": "", "Twitter": "" }).</li>
</ul>

<h3>Recommendation</h3>
<p><strong>Mixtral</strong> is recommended for more detailed and structured information, especially for phone numbers and social media. <strong>Llama</strong> is suitable for simpler outputs where detailed structure is less critical.</p>
