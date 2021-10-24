# Resources
One need prepare two resource file:

1. The Faiss index. The file name should end with ".index";
2. The ID-mapping file, where the i-th line is corresponding to the document ID of the i-th vector. The file name should end with ".txt".

**NOTICE**: Faiss-instant will only load from one file ending with ".txt" and one file ending with ".index". So please avoid storing multiple of them.