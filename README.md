# Lzg-Jlu

## AVL-DRIVE 风格开源近似算法

面向驾驶性/电动回收/ADAS/舒适性方法学对标的四类近似评价算法（**非** AVL 闭源实现）：

- Tip-in 纵向驾驶性
- 回收与机械制动交接（VIORE 风格）
- ACC Cut-in 感知安全/舒适
- 垂向 Ride 舒适

详见 [`avl_drive_approx/METHOD.md`](avl_drive_approx/METHOD.md)。

### 运行示例

```bash
python -m avl_drive_approx.examples.run_demo
python -m pytest tests -q
```
