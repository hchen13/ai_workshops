import multiprocessing
import os
import threading
from multiprocessing.pool import ThreadPool

import requests
from selenium import webdriver

base_dir = os.path.join(os.path.expanduser('~'), 'datasets', 'artworks')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

genres = [
    ('portrait',100),
    ('landscape',100),
    ('genre-painting',100),
    ('abstract',100),
    ('religious-painting',100),
    ('cityscape',100),
    ('figurative',50),
    ('still-life',50),
    ('symbolic-painting',50),
    ('nude-painting-nu',50),
    ('mythological-painting',30),
    ('marina',30),
    ('flower-painting',30),
    ('animal-painting',30)
]


def init_driver(headless=False):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('headless')
    return webdriver.Chrome(options=options)


def get_links(genre, page_num):
    url = "https://www.wikiart.org/en/paintings-by-genre/{}/{}".format(genre, page_num)
    image_urls = []
    chrome = init_driver(headless=True)
    chrome.set_page_load_timeout(30)
    chrome.implicitly_wait(30)
    try:
        chrome.get(url)
        images = chrome.find_elements_by_tag_name('img')
        for image in images:
            image_urls.append(image.get_attribute('src'))
        print("[info] page {} image urls recording complete.".format(page_num), flush=True)
    except:
        print('[warning] open page {} failed, skipping...'.format(page_num), flush=True)
    chrome.close()
    return image_urls


def valid_url(url):
    if not isinstance(url, str):
        return False
    if not url.startswith('https://'):
        return False
    return True


def scrape(genre, pages):
    folder = os.path.join(base_dir, genre)
    if not os.path.exists(folder):
        os.makedirs(folder)
    threads_count = multiprocessing.cpu_count()
    print("[info] finding images of genre [{}] with {} threads...".format(genre, threads_count))

    pool = ThreadPool(threads_count)
    results = pool.starmap(get_links, zip([genre] * pages, list(range(1, pages + 1))))
    pool.close()

    urls = sum(results, [])
    urls = list(filter(valid_url, urls))
    num_images = len(urls)

    print("[info] {} images found, saving URLs into text file...".format(num_images))
    with open('tmp/{}_urls.txt'.format(genre), 'w') as fout:
        for url in urls:
            fout.write(url + '\n')


def download_batch(start, end, genre):
    global count, count_invalid, record
    url_file = 'tmp/{}_urls.txt'.format(genre)

    with open(url_file, 'r') as f:
        for i, url in enumerate(f.readlines()[start:end]):
            url = url.rstrip('\n')
            fullname = os.path.join(base_dir, genre, '{}.jpg'.format(start + i))
            try:
                image_content = requests.get(url, timeout=10).content
                with open(fullname, 'wb') as handler:
                    handler.write(image_content)
                record += 1
                print('\r[info] progress: {}/{} images have been downloaded'.format(record, count), flush=True, end='')
            except requests.Timeout:
                count_invalid += 1


def download_images(genre, num_threads=multiprocessing.cpu_count()):
    folder = os.path.join(base_dir, genre)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    url_file = 'tmp/{}_urls.txt'.format(genre)
    global count, count_invalid, record
    count = 0
    count_invalid = 0
    record = 0
    if not os.path.exists(url_file):
        return
    with open(url_file, 'r') as f:
        count += len(f.readlines())
    part = int(count / num_threads)
    print("[info] batch downloading [{}] images with {} threads...".format(genre, num_threads))
    thread_list = []
    for i in range(num_threads):
        end = count if i == num_threads - 1 else (i + 1) * part
        t = threading.Thread(target=download_batch, kwargs={'start': i * part, 'end': end, 'genre': genre})
        t.setDaemon(True)
        thread_list.append(t)
        t.start()

    for i in range(num_threads):
        try:
            while thread_list[i].isAlive():
                pass
        except KeyboardInterrupt:
            break

    print("\nGenre [{}] finished, success/invalid = {}/{}\n\n".format(genre, count - count_invalid, count_invalid))


if __name__ == '__main__':

    for genre, pages in genres:
        scrape(genre, pages)

    for genre, _ in genres:
        download_images(genre, num_threads=multiprocessing.cpu_count() * 4)
