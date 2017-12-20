### FlapPytorchBird

`python dqn.py`

### Quick fix for Windows user in Visdom
The problem is that windows doesn't allow "?" in filenames. But Visdom library contains this problem.

A quick fix for windows users would be to delete the corresponding entry in ext_files in server.py

```python
%sfonts/glyphicons-halflings-regular.eot?#iefix' % bb:
'glyphicons-halflings-regular.eot?#iefix',

```
(lines 568-569)  

### Visualization with Visdom
`python -m visdom.server`

See the result : [URL](localhost:8097)

### RESULT
##### training log
[](/assets/result/result_1.png)
[](/assets/result/result_2.png)

##### result video
[![FlapPyTorchBird](/assets/result/result_3.png)](https://www.youtube.com/watch?v=WzAMlcJiYNQ "Flappy bird after Machine Learning")
