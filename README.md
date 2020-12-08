# **Co-Attentive Representation Learning for Web Services Classification**

The rapid adoption of services related technologies, such as cloud computing, has lead to the explosive growth of web services. Automated service classification that groups web services by similar functionality is a widely used technique to facilitate the management and discovery of web services within a large-scale repository. The existing service classification approaches primarily focus on learning the isolated representations of service features but ignored their internal semantic correlations. To address the aforementioned issue, we propose a novel deep neural network with the co-attentive representation learning mechanism for effectively classifying services by learning interdependent characteristics of service without feature engineering. Specifically, we propose a service data augmentation mechanism by extracting informative words from the service description using information gain theory. Such a mechanism can learn a correlation matrix among embedded augmented data and description, thereby obtaining their interdependent semantic correlation representations for service classification. We evaluate the effectiveness of our proposed approach by comprehensive experiments based on a real-world dataset collected from ProgrammableWeb, which includes 10,943 web services. Compared with seven web service classification baselines based on CNN, LSTM, Recurrent-CNN, C-LSTM, BLSTM, ServeNet and ServeNet-BERT, the proposed approach can achieve an improvement of 5.66%-172.21% in the F-measure of web service classification.



# Results

| Model         | $$A_{top1}$$ | $$A_{top5}$$ | Precision | Recall    | F-measure |
| ------------- | ------------ | ------------ | --------- | --------- | --------- |
| CNN           | 0.295        | 0.569        | 0.296     | 0.240     | 0.252     |
| LSTM          | 0.510        | 0.789        | 0.477     | 0.410     | 0.403     |
| RCNN          | 0.597        | 0.857        | 0.608     | 0.529     | 0.543     |
| CLSTM         | 0.612        | 0.851        | 0.595     | 0.585     | 0.578     |
| BLSTM         | 0.619        | 0.853        | 0.636     | 0.576     | 0.587     |
| ServeNet      | 0.631        | 0.874        | 0.631     | 0.602     | 0.608     |
| ServeNet-BERT | 0.681        | **0.905**    | 0.668     | 0.653     | 0.654     |
| Ours          | **0.715**    | 0.890        | **0.703** | **0.689** | **0.691** |



# Run

**train** python main.py --mode train

**eval**  python main.py --mode eval