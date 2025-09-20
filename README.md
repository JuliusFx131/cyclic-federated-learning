# Understanding Cyclic Federated Training with XGBoost

This repo implements a form of **federated learning** using **XGBoost** in a *cyclic training style*. Let’s break it down.

**Traditional Federated Learning (FedAvg)**  
- Clients train locally and send updates to a **central server**, which averages them into a global model.  
- ✅ Scalable, parallel training, and widely used in practice.  
- ⚠️ Requires a server and can struggle with very different (non-IID) data.  
*(McMahan et al., 2017 — [FedAvg Paper](https://arxiv.org/abs/1602.05629))*

**Cyclic Federated Learning (CFL)**  
- The global model is passed **client-to-client in a cycle**, each site updating before handing it on.  
- ✅ No server required, simpler communication, fits decentralized settings.  
- ⚠️ Sequential (slower) and client order can influence results.  
*(Pengyu  et al., 2021 — [Cyclic FL](https://arxiv.org/abs/2301.12193))*

---

## 1. Setup

```python
global_bst = None
print("\nStarting cyclic federated training:")
```

- `global_bst` stores the **global model** (an XGBoost booster).  
- At the start, it’s `None`, meaning training begins from scratch.

---

## 2. Global Rounds

```python
for gr in range(1, CONFIG["NUM_GLOBAL_ROUNDS"] + 1):
    print(f"\n=== Global round {gr}/{CONFIG['NUM_GLOBAL_ROUNDS']} ===")
```

- Federated learning proceeds in **global rounds**.  
- In each round, the global model is trained across all clients.

---

## 3. Local Training (per client)

```python
for site in SITES:
    client = site_data[site]
    print(f"Training on client '{site}' ({client['n_train']} samples)...")
    global_bst = xgb.train(
        xgb_params,
        client["dtrain"],
        num_boost_round=CONFIG["NUM_LOCAL_ROUNDS"],
        xgb_model=global_bst,
        evals=[(client["dval"], "val")],
        verbose_eval=5,
    )
```

- Each **site** (client) provides its own training and validation data:  
  - `client["dtrain"]` → local training set  
  - `client["dval"]` → local validation set  
- Training happens **sequentially**: the global model is updated client by client.  
- `xgb_model=global_bst` ensures that each client starts from the **current global model**.  
- After local training, the updated model replaces `global_bst`.

This sequential handoff is what makes it **cyclic federated training**.

---

## 4. Global Evaluation

```python
pooled_preds = global_bst.predict(pooled_val)
true_labels = pooled_val.get_label()
```

- After each global round, the model is evaluated on a **pooled validation set** (all sites combined).  
- This measures overall performance and generalization.

*(The accuracy computation is commented out in the snippet, but can be re-enabled to track progress.)*

---

## 🔑 Summary

- **What’s happening:**  
  The code simulates **federated learning with XGBoost** by passing the global model sequentially through each client.  

- **How it works:**  
  - Each client fine-tunes the global model.  
  - The updated model is handed to the next client.  
  - After all clients finish, we complete one **global round**.  

- **Key difference from FedAvg:**  
  - In **FedAvg**, models are trained in parallel on each client and then averaged.  
  - In **cyclic training**, the model is trained **sequentially** across clients.  

- **Outcome:**  
  The global model gradually incorporates knowledge from all clients **without centralizing their raw data**.
