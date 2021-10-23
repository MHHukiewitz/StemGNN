import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, nodes, multi_layer, backcast=False):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.nodes = nodes
        self.does_backcast = backcast
        self.multi = multi_layer
        self.conv_kernel = (4, 3)
        self.conv_real = nn.Conv2d(self.nodes, self.nodes, self.conv_kernel, padding=(0, 1))
        self.rev_conv_real = nn.ConvTranspose2d(self.nodes, self.nodes, self.conv_kernel, padding=(0, 1))
        self.conv_imag = nn.Conv2d(self.nodes, self.nodes, self.conv_kernel, padding=(0, 1))
        self.rev_conv_imag = nn.ConvTranspose2d(self.nodes, self.nodes, self.conv_kernel, padding=(0, 1))
        self.weight = nn.Parameter(
            torch.Tensor(1, self.multi * self.time_step, self.multi * self.time_step))
        # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        self.selu = torch.nn.SELU()
        if self.does_backcast:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.GLUs = nn.ModuleList()
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step, self.time_step * self.multi))
                self.GLUs.append(GLU(self.time_step, self.time_step * self.multi))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.multi, self.time_step * self.multi))
                self.GLUs.append(GLU(self.time_step * self.multi, self.time_step * self.multi))
            else:
                self.GLUs.append(GLU(self.time_step * self.multi, self.time_step * self.multi))
                self.GLUs.append(GLU(self.time_step * self.multi, self.time_step * self.multi))

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, k, node_cnt, time_step)
        ffted = torch.fft.fft(input, norm="ortho")
        ffted = ffted.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, k, time_step)
        real = self.conv_real(ffted.real).reshape(batch_size, node_cnt, -1)
        imag = self.conv_imag(ffted.imag).reshape(batch_size, node_cnt, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            imag = self.GLUs[2 * i + 1](imag)
        #real = self.rev_conv_real(real.reshape(batch_size, node_cnt, 1, -1))
        #imag = self.rev_conv_imag(imag.reshape(batch_size, node_cnt, 1, -1))
        #real = real.reshape(batch_size, node_cnt, -1).permute(0, 2, 1, 3).contiguous()
        #imag = imag.reshape(batch_size, node_cnt, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.complex(real, imag)
        iffted = torch.fft.ifft(time_step_as_inner, norm="ortho").real
        return iffted

    def forward(self, x, cheb_polynomial):
        cheb_polynomial = cheb_polynomial.unsqueeze(1)
        gfted = torch.matmul(cheb_polynomial, x.unsqueeze(1))
        gconv_input = self.spe_seq_cell(gfted)
        igfted = torch.matmul(gconv_input, self.weight)

        forecast_source = torch.sigmoid(self.forecast(igfted))
        forecast = self.forecast_result(forecast_source)
        if self.does_backcast:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short).unsqueeze(1)
        else:
            backcast_source = None
        return forecast, backcast_source


class Model(nn.Module):
    def __init__(self, nodes, stack_cnt, window_size, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.nodes = nodes
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.window_size = window_size
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.nodes, 1)))
        nn.init.kaiming_uniform_(self.weight_key.data, nonlinearity='leaky_relu')
        self.weight_query = nn.Parameter(torch.zeros(size=(self.nodes, 1)))
        nn.init.kaiming_uniform_(self.weight_query.data, nonlinearity='leaky_relu')
        self.GRU = nn.GRU(self.window_size, self.nodes)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.window_size, self.nodes, self.multi_layer, backcast=(i < (self.stack_cnt - 1)))
             for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.window_size), int(self.window_size)),
            nn.LeakyReLU(self.alpha),
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.window_size), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention
