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

