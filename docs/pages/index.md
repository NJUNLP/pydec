---
layout: page
title: PyDec Documentation Home
permalink: /
---

# Welcome to PyDec Documentation

PyDec is a linear decomposition toolkit for neural network based on [PyTorch](https://pytorch.org/), which can decompose the tensor in the forward process into given components with a small amount of code. The result of decomposition can be applied to tasks such as attribution analysis.

# Quick Links
<div class="section-index">
    <hr class="panel-line">
    {% for post in site.docs %}
        {% assign path_segments = post.url | split: "/" %}
            {% if path_segments[2] == "Archive" %}
                <div class="entry">
                <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
                <p>{{ post.description }}</p>
                <p>{{ post.url }}</p>
                <p>xxx{{ path_segments[2] }}xxx</p>
                </div>
            {% endif %}
    {% endfor %}
</div>


Would you like to request a feature or contribute?
[Open an issue]({{ site.repo }}/issues)
