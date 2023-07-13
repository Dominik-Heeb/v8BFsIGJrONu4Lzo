# Term Deposits

Apziva project #2<br>
2023 06 28ff

__Summary:__
* A bank tries to convince their clients to __subscribe to a product: a term deposit__. This subscription, either "yes" or "no", is the ML target.
* Alongside their __marketing calls__, the callers collected __13 descriptors__ for each client, socio-economic factors as much as conversation-related data.
* As only about 7% of the clients finally subscribed to the product the dataset was __strongly imbalanced__. This stronlgy suggested to use __F1 as the metric__ to be used while evaluating the models, rather than the otherwise more common accuracy.
* Some effort was invested to identify the __start and the end of the marketing campaign__. With high probability, the marketing campaign startet mid-October 2014 and ended in August 2015. This finding allowed for interesting feature engineering.
* When using all variables, an F1 score of 87% is reached.
* I my-self am skeptical that using all variables is useful, as some of the variables are rather target-like than causal to the target, i.e. the client's decision to subscribe to the new product.