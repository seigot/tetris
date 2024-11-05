device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@@ -5,0 +6,1 @@
model.to(device)
@@ -10,1 +11,1 @@
-tensor = torch.tensor(data)
tensor = torch.tensor(data, device=device)
@@ -15,1 +16,1 @@
-tensor = tensor
tensor = tensor.to(device)
@@ -20,1 +21,1 @@
-result = tensor1 + tensor2
result = tensor1.to(device) + tensor2.to(device)
