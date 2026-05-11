import torch
import math
import matplotlib.pyplot as plt

def poly_regression():
    print("--- 线性多项式拟合 与 优化器对比 ---")
    
    # 1. 准备数据: y = sin(x) 带有少量噪声，区间 [-pi, pi]
    x = torch.linspace(-math.pi, math.pi, 2000).unsqueeze(-1)
    y_true = torch.sin(x) + torch.randn(2000, 1) * 0.1
    
    # 手动实现不同的优化算法 (GD, SGD, 或者 Mini-Batch SGD)。
    
    print(f"\n开始训练...")
    
    # 2. 初始化参数 
    # 你可以自己决定多项式的最高次数，例如使用 3 次多项式 a + b*x + c*x^2 + d*x^3
    ### TODO: 初始化多项式参数 (如 a, b, c, d), 记得设置 requires_grad=True
    
    ### END TODO
    
    # 提示：SGD 学习率如果和 GD 一样大会导致梯度爆炸，可能需要动态调整 learning_rate 或 epochs
    learning_rate = 1e-5
    epochs = 1000
    
    losses = []
    for epoch in range(epochs):
        indices = torch.randperm(len(x))
        x_shuffled = x[indices]
        y_shuffled = y_true[indices]
        
        epoch_loss = 0.0
        
        # --- 核心训练逻辑 ---
        # 提示：你需要自己决定如何遍历数据 (一次送入全部、一次送入一个、还是一次送入一批)
        # 在这里执行 前向传播 -> 计算Loss -> 反向传播 -> 更新权重。
        
        ### TODO: 实现数据遍历与更新逻辑
        
        ### END TODO
        
        # 记录每个 Epoch 的平均 Loss
        if epoch_loss > 0:
            epoch_loss /= len(x)
            losses.append(epoch_loss)
            if epoch % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
    # 绘图展示拟合结果
    try:
        with torch.no_grad():
            ### TODO: 用训练好的参数代入全集 x 求解 y_final_pred 绘图
            y_final_pred = None
            ### END TODO
            
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Loss Curve')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(x.numpy(), y_true.numpy(), s=1, alpha=0.3, label='Data')
        if y_final_pred is not None:
            plt.plot(x.numpy(), y_final_pred.numpy(), color='red', linewidth=2, label=f'Fit')
        plt.title(f"Polynomial Fit")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("poly_regression_result.png")
        print("\n=> 训练结果已保存为 poly_regression_result.png！如果右侧只有点没有红线，请检查是否完成功能")
    except:
        pass

if __name__ == '__main__':
    poly_regression()
