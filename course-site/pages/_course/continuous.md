---
title: 'miniLLM 课程连续阅读'
layout: course_page
permalink: '/course/continuous/'
type: '连续阅读'
sidebar:
  nav: course-continuous
source_path: 'generated:continuous'
---

{% for page in site.course %}
  {% if page.source_path == "course/README.md" %}
<section id="course-home" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/00-orientation.md" %}
<section id="chapter-0" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/practice/README.md" %}
<section id="lab-guide" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch01-step0-tensor.md" %}
<section id="ch01-step0-tensor" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch02-step1-tokenizer.md" %}
<section id="ch02-step1-tokenizer" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch03-step2-embedding.md" %}
<section id="ch03-step2-embedding" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch04-step3-attention.md" %}
<section id="ch04-step3-attention" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch05-step4-transformer.md" %}
<section id="ch05-step4-transformer" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch06-step5-gpt.md" %}
<section id="ch06-step5-gpt" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch07-step6-training.md" %}
<section id="ch07-step6-training" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch08-step7-generation.md" %}
<section id="ch08-step7-generation" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch09-step8-chat.md" %}
<section id="ch09-step8-chat" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch10-step9-http.md" %}
<section id="ch10-step9-http" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch11-step10-kv-cache.md" %}
<section id="ch11-step10-kv-cache" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch12-step11-bpe.md" %}
<section id="ch12-step11-bpe" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

{% for page in site.course %}
  {% if page.source_path == "course/chapters/ch13-end-to-end.md" %}
<section id="ch13-end-to-end" class="chapter-section" markdown="1">
## {{ page.title }}
{{ page.content }}
</section>
  {% endif %}
{% endfor %}

<script>
document.addEventListener('DOMContentLoaded', function() {
  const sections = document.querySelectorAll('.chapter-section');
  const navLinks = document.querySelectorAll('.sidebar a[href*="#"]');
  function updateActiveLink() {
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop;
      if (window.pageYOffset >= sectionTop - 120) {
        current = section.getAttribute('id');
      }
    });
    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href').includes('#' + current)) {
        link.classList.add('active');
      }
    });
  }
  window.addEventListener('scroll', updateActiveLink);
  updateActiveLink();
});
</script>
