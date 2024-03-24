# Understanding and Mitigation of Noise in Crowd Sourced Relation Extraction Dataset
Text is one of the main sources of transmission of information between humans. It varies from personal chats to corporate e-mails, legal documents to scientific research articles, social media posts to blogs, and news bulletins to public announcements. The amount of such unstructured text in the open world is rising at an exponential rate. However, only a fragment of this information is available as knowledge for computer algorithms. Natural Language Processing (NLP) tasks such as Information Extraction (IE) and its sub-task Relation Extraction (RE) / Relation Classification (RC) can significantly improve the conversion from unstructured text to structured knowledge. Nonetheless, RE/RC is largely restricted due to the absence of high-quality datasets for training data-hungry deep neural models, which have shown excellent performance in other NLP tasks.

The overarching objective of this dissertation is to explore unconventional ways of improving large RC datasets and learning from them. Existing large RC datasets have few relations labels, ignore relations between relations, and are noisy and imbalanced. Creating a new dataset is not an optimal solution as it is a time- and cost-intensive process. The primary focus of this thesis is to analyze noise present in existing large-scale RC datasets and propose automated methods to mitigate some of the noise. In particular, work is focused on three main objectives, *(i)* characterizing the noise present in the dataset; *(ii)* exploring automatic and cost-sensitive approaches to reduce noise from the RC dataset; and *(iii)* analyzing the cost of reannotating them. 

To this end, this dissertation makes three major contributions toward improving the RC from a large crowd-sourced dataset TACRED. The \textit{first} work focuses on exploring the use of the relation between relation labels for reducing noise and improving RC models. Our preliminary analysis as well as some contemporary studies indicate that several incorrect relation labels can be identified by examining the corresponding `subject_entity` and `object_entity`. Based on this observation, we build a taxonomical relation hierarchy (TRH) from multiple KBs. We used it as a template for creating a similar TRH for TACRED which is then used for exploring noise in positive instances and incorporating hierarchical distance between relation labels in RC models.

For our *second* contribution, we did a comprehensive evaluation of noise in the TACRED. All our analyses are based on SOTA RC models' predictions and performance. Following our findings, we investigate automated and cost-sensitive strategies for reducing noisy instances based on the nearest neighbors of examples with false-negative predictions and examples from a cleaner subset of TACRED. Empirical results have shown improved performance on the newly generated datasets.

In our *third* and final contribution, we utilize relation hierarchy for budget-sensitive reannotation of TACRED. We introduce the concept of a reannotation budget to provide flexibility on how much data to reannotate. We also proposed two strategies for selecting data points for reannotation. We performed extensive experiments using the popular RC dataset TACRED. We have shown that our reannotation strategies are novel and more efficient when compared with the existing approaches. Our experiments suggest that the reported performance of existing RC models on the noisy dataset is inflated. Furthermore, we show that significant model performance improvement can be achieved by reannotating only a part of the training data.



## References

1. (Parekh, Akshay, Ashish Anand, and Amit Awekar. "Improving Relation Classification Using Relation Hierarchy." International Conference on Applications of Natural Language to Information Systems. Cham: Springer International Publishing, 2022.) [https://link.springer.com/chapter/10.1007/978-3-031-08473-7_29]

## Citation
```
{ link : http://hdl.handle.net/10603/459884
Title of Thesis : Understanding and Mitigation of Noise in Crowd Sourced Relation Classification Dataset
Name of the Researcher : Parekh, Akshay
Name of the Guide : Awekar, Amit and Anand, Ashish
Completed Year : 2023
Abstract : Relation classification (RC), a task of classifying the relation between a given pair of entities in a sentence to a relation label is fundamental to IE systems. The identified structured triple (subject_entity, relation, object_entity) from the unstructured text can vastly help in knowledge base completion. This organized relational knowledge can further be used for other downstream tasks like question-answering, and common-sense reasoning. A large RC dataset TACRED has been widely used for benchmarking modern deep neural models. However, RC at a large scale is restricted mainly due to the presence of noise in the training dataset. Hence, the performance of such advanced deep neural models, which have shown excellent improvement on other NLP tasks, has been held back for RC.
Name of the Department : Department of Computer Science and Engineering
Name of the University : Indian Institute of Technology Guwahati }
```