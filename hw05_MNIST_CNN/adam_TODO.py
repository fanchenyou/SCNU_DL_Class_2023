import torch


#####################
## read docs first ##
#####################
# https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

class MyADAMOptimizer(torch.optim.Optimizer):
    """
    implements ADAM Algorithm, as a preceding step.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(MyADAMOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.
        Simplify the official implementation
        https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
        """
        loss = None
        for group in self.param_groups:

            for p in group['params']:
                #############################
                ## DO NOT Modify Code here ##
                #############################
                # g_t
                g_t = p.grad.data
                # history state
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # m_t -- Momentum (Exponential MA of gradients)
                    state['m_t_1'] = torch.zeros_like(p.data)

                    # v_t -- Denominator. (Exponential MA of squared gradients).
                    state['v_t_1'] = torch.zeros_like(p.data)

                # m_(t-1), v_(t-1) are moving average
                m_t_1, v_t_1 = state['m_t_1'], state['v_t_1']

                b1, b2 = group['betas']
                state['step'] += 1

                # Add weight decay
                if group['weight_decay'] != 0:
                    g_t = g_t.add(group['weight_decay'], p.data)
                ##############################
                ## DO NOT Modify Code Above ##
                ##############################

                #######################################
                ## TODO: Modify the code below !!!!! ##
                #######################################
                # you should modify this entire code block to utilize
                # m, v, m_hat, v_hat, step_no and other necessary variables such as b1, b2
                # update m_t, TODO modify
                m_t = m_t_1
                # update v_t, TODO  modify
                v_t = v_t_1
                # update m_hat_t, TODO modify
                m_hat_t = None
                # update v_hat_t, TODO modify
                v_hat_t = None
                # update parameter with m_t, v_t, learning rate group['lr']
                # modify this, currently it's just SGD
                p.data = p.data - g_t * group['lr']
                ####################################
                ## Stop modification of code here ##
                ####################################

                # Save state
                state['m_t_1'], state['v_t_1'] = m_t, v_t

        return loss
