# Adaptive Learning Rate Clipping Stabilizes Learning

A simple implementation of [Adaptive Learning Rate Clipping](https://arxiv.org/abs/1906.09060) in Pytorch.

Please see the paper (linked above) or the accompanying code [for TensorFlow](https://github.com/Jeffrey-Ede/ALRC) for a succinct and well-written paper and repo implementation. Below is a (very) simple example on how to use this PyTorch version:

```python
model = Net()
loss = nn.MSELoss()
clipper = ALRC()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    output = model(input)
    loss = loss_fn(output, target)
    loss = clipper.clip(loss)

    loss.backward()
    optimizer.step()
```
