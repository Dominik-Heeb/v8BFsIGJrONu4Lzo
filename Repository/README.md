# Term Deposits

Apziva project #2<br>
2023 06 28ff

__Summary:__
* A bank tries to convince their clients to __subscribe to a product: a term deposit__. This subscription, either "yes" or "no", is the ML target.
* Alongside their __marketing calls__, the callers collected __13 descriptors__ for each client, socio-economic factors as much as conversation-related data.
* As only about 7% of the clients finally subscribed to the product the dataset was __strongly imbalanced__. This stronlgy suggested to use __F1 as the metric__ to be used while evaluating the models, rather than the otherwise more common accuracy.
* Some effort was invested to identify the __start and the end of the marketing campaign__. With high probability, the marketing campaign startet mid-October 2014 and ended in August 2015. This finding allowed for interesting feature engineering.
* From four models evaluated, ridge regression yielded the best __F1 value: 0.451__.
* Surprisingly, __socio-economic features, loans and balance__ seem __not__ to be __relevant__ to the clients' readiness to subscribe to the product. Instead, subscriptions were more likely __the longer the marketing campaign lasted__, but not if the prolonged into the __last two months__ of the marketing campaign. Also a __long last talk__ points to a probable subscription.

__To-do's__
* Re-do the models only with features directly related to the client:
	* socio-economic factros, loan, default. 
	* i.e. without duration of last call, number of contacts, date of last call...
	* purpose: finding customer segments to prioritize.
* No refactoring done yet (folder "src"). Refactoring will be done after re-doing models (see above) and feedback by SSM.