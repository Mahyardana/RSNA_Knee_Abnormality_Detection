using System;
using System.Collections.Generic;
using System.Runtime.InteropServices.Swift;
using System.Text;
using System.Xml.Schema;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Picturescategorizer_ConsoleApp1
{
    internal class TransformerBlock : nn.Module<Tensor, Tensor>
    {
        public TransformerBlock(int dim, int heads = 8,double mlp_ratio=4.0, double dropout = 0.0) : base("TransformerBlock")
        {
            norm1 = nn.LayerNorm(dim);
            attn = new MHSA(dim, heads, dropout);
            norm2 = nn.LayerNorm(dim);
            mlp = nn.Sequential(
                nn.Linear(dim,(int)((double)dim*mlp_ratio)),
                nn.GELU(),nn.Dropout(dropout),
                nn.Linear((int)((double)dim*mlp_ratio),dim),
                nn.Dropout(dropout)
                );
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using DisposeScope disposeScope = torch.NewDisposeScope();
            var output = input + attn.call(norm1.call(input));
            output = output + mlp.call(norm2.call(output));
            disposeScope.MoveToOuter(output);
            return output;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                attn.Dispose();
                norm1.Dispose();
                norm2.Dispose();
                mlp.Dispose();
            }
            base.Dispose();
        }
        private int heads;
        private double scale;
        private MHSA attn;
        private nn.Module<Tensor, Tensor> norm1;
        private nn.Module<Tensor, Tensor> norm2;
        private nn.Module<Tensor, Tensor> mlp;
    }
    internal class MHSA : nn.Module<Tensor, Tensor>
    {
        public MHSA(int dim, int heads = 8, double dropout = 0.0) : base("MHSA")
        {
            this.heads = heads;
            scale = Math.Pow(((int)dim / heads), -0.5);
            qkv = nn.Linear(dim, dim * 3);
            attDropout = nn.Dropout(dropout);
            proj = nn.Linear(dim, dim);
            projDropout = nn.Dropout(dropout);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using DisposeScope disposeScope = torch.NewDisposeScope();
            var B = input.shape[0];
            var N = input.shape[1];
            var D = input.shape[2];
            var qkvout = qkv.call(input).chunk(3, dim: -1);
            var q = qkvout[0];
            var k = qkvout[1];
            var v = qkvout[2];
            q = q.reshape(new[] { B, N, heads, -1 }).transpose(1, 2);
            k = k.reshape(new[] { B, N, heads, -1 }).transpose(1, 2);
            v = v.reshape(new[] { B, N, heads, -1 }).transpose(1, 2);
            var attn = torch.matmul(q, k.transpose(-2, -1)) * scale;
            attn = attDropout.call(attn.softmax(dim: -1));
            input = torch.matmul(attn, v).transpose(1, 2).reshape(new [] { B, N, D });
            var output = projDropout.call(proj.call(input));
            disposeScope.MoveToOuter(output);
            return output;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                qkv.Dispose();
                attDropout.Dispose();
                proj.Dispose();
                projDropout.Dispose();
            }
            base.Dispose();
        }
        private int heads;
        private double scale;
        private nn.Module<Tensor, Tensor> qkv;
        private nn.Module<Tensor, Tensor> attDropout;
        private nn.Module<Tensor, Tensor> proj;
        private nn.Module<Tensor, Tensor> projDropout;
    }
    internal class ConvNet : nn.Module<Tensor, Tensor>
    {
        public ConvNet() : base("ConvNet")
        {

            var tbasemodel = torchvision.models.resnet18();
            var modules = tbasemodel.modules();
            var moduleslist = new List<nn.Module<Tensor, Tensor>>();
            var children = tbasemodel.named_children().ToDictionary();
            //moduleslist.Add(nn.Conv2d(3, 32, 5, padding: 2));
            //moduleslist.Add(nn.ReLU());
            //moduleslist.Add(nn.MaxPool2d(2));
            //moduleslist.Add(nn.BatchNorm2d(32));
            //moduleslist.Add(nn.Conv2d(32, 64, 5, padding: 2));
            //moduleslist.Add(nn.ReLU());
            //moduleslist.Add(nn.MaxPool2d(2));
            //moduleslist.Add(nn.BatchNorm2d(64));
            //moduleslist.Add(nn.Conv2d(64, 128, 5, padding: 2));
            //moduleslist.Add(nn.ReLU());
            //moduleslist.Add(nn.MaxPool2d(2));
            //moduleslist.Add(nn.BatchNorm2d(128));
            //moduleslist.Add(nn.Conv2d(128, 256, 5, padding: 2));
            //moduleslist.Add(nn.ReLU());
            //moduleslist.Add(nn.MaxPool2d(2));
            //moduleslist.Add(nn.BatchNorm2d(256));
            //moduleslist.Add(nn.Conv2d(256, 512, 5, padding: 2));
            //moduleslist.Add(nn.ReLU());
            //moduleslist.Add(nn.MaxPool2d(2));
            //moduleslist.Add(nn.BatchNorm2d(512));
            //moduleslist.Add(nn.Conv2d(512, 1024, 5, padding: 2));
            //moduleslist.Add(nn.ReLU());
            //moduleslist.Add(nn.MaxPool2d(2));
            //moduleslist.Add(nn.BatchNorm2d(1024));
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["conv1"]);
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["bn1"]);
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["relu"]);
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["maxpool"]);
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["layer1"]);
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["layer2"]);
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["layer3"]);
            moduleslist.Add((nn.Module<Tensor, Tensor>)children["layer4"]);
            //moduleslist.Add((nn.Module<Tensor, Tensor>)children["avgpool"]);
            seq1 = torch.nn.Sequential(moduleslist);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using DisposeScope disposeScope = torch.NewDisposeScope();
            var toutput = seq1.call(input);
            disposeScope.MoveToOuter(toutput);
            return toutput;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                seq1.Dispose();
            }
            base.Dispose();
        }
        private nn.Module<Tensor, Tensor> seq1;
    }
    internal class ViT : nn.Module<Tensor, Tensor>
    {
        public ViT(int inputdim = 1024, int num_classes = 1000, int embeddim = 768, int depth = 6, int heads = 8, double mlpratio = 0.4, double dropoutratio = 0.1, int max_tokens = 50) : base("ViT")
        {

            //moduleslist.Add(nn.AdaptiveAvgPool2d((1,1)));
            patchProjection = nn.Conv2d(inputdim, embeddim, kernel_size: 1);
            //moduleslist.Add(patchProjection);
            flattenAtt = nn.Flatten(2);
            //moduleslist.Add(transformerEncoder);
            //transformerEncoder.forward(,)
            //moduleslist.Add(transformerEncoder);
            // Transformer Encoder
            cls = nn.Parameter(torch.zeros(1, 1, embeddim));
            pos = nn.Parameter(torch.randn(1, max_tokens, embeddim));
            var moduleslist = new List<nn.Module<Tensor, Tensor>>();
            for (int i = 0; i < depth; i++)
            {
                moduleslist.Add(new TransformerBlock(embeddim, heads, mlpratio, dropoutratio));
            }
            blocks = nn.Sequential(moduleslist);
            norm = nn.LayerNorm(embeddim);
            headAtt = nn.Linear(embeddim, num_classes);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using DisposeScope disposeScope = torch.NewDisposeScope();
            Tensor? toutput = null;
            toutput = patchProjection.call(input);
            toutput = flattenAtt.call(toutput).transpose(1, 2);
            var B = toutput.shape[0];
            var L = toutput.shape[1];
            clstensor = cls.expand(new[] { B, -1, -1 });
            toutput = torch.cat(new[] { clstensor, toutput }, 1);
            toutput = toutput + pos[.., 0..(int)(L + 1)];
            toutput = blocks.call(toutput);
            toutput = norm.call(toutput)[.., 0];
            toutput = headAtt.call(toutput);
            disposeScope.MoveToOuter(toutput);
            return toutput;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                patchProjection.Dispose();
                flattenAtt.Dispose();
                norm.Dispose();
                headAtt.Dispose();
                blocks.Dispose();
            }
            base.Dispose();
        }
        private nn.Module<Tensor, Tensor> patchProjection;
        private nn.Module<Tensor, Tensor> flattenAtt;
        private nn.Module<Tensor, Tensor> blocks;
        private nn.Module<Tensor, Tensor> norm;
        private nn.Module<Tensor, Tensor> headAtt;
        private Parameter cls;
        private Tensor clstensor;
        private Parameter pos;
    }
    internal class ViTHybrid : nn.Module<Tensor, Tensor>
    {
        public ViTHybrid(int num_classes = 1000) : base("ViTHybrid")
        {
            cnn = new ConvNet();
            vit = new ViT(512, num_classes, 768, 12, 12);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using DisposeScope disposeScope = torch.NewDisposeScope();
            Tensor? toutput = null;
            toutput = cnn.call(input);
            toutput = vit.call(toutput);
            disposeScope.MoveToOuter(toutput);
            return toutput;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                vit.Dispose();
                cnn.Dispose();
            }
            base.Dispose();
        }
        private nn.Module<Tensor, Tensor> cnn;
        private nn.Module<Tensor, Tensor> vit;
    }
}
