# 十大经典的排序算法

```c++
#ifndef SORT_ALGO_H
#define SORT_ALGO_H

#include <vector>
using std::vector;
using std::swap;

// 1. Bubble Sort
// Time complexity: O(n^2)
// Space complexity: O(1)
void bubble_sort(vector<int>& nums)
{
  bool sorted = false;
  for (int i = 0; i < nums.size() && !sorted; ++i)
  {
    sorted = true;
    for (int j = 1; j < nums.size() - i; ++j)
    {
      if (nums[j] < nums[j - 1])
      {
        swap(nums[j], nums[j - 1]);
        sorted = false;
      }
    }
  }
}

// 2. Insertion Sort
// Time complexity: O(n^2)
// Space complexity: O(1)
void insertion_sort(vector<int>& nums)
{
  for (int i = 1; i < nums.size(); ++i)
  {
    for (int j = i; j > 0 && nums[j] < nums[j - 1]; --j)
      swap(nums[j], nums[j - 1]);
  }
}

// 3. Selection Sort
// Time complexity: O(n^2)
// Space complexity: O(1)
void selection_sort(vector<int>& nums)
{
  for (int i = 0; i < nums.size(); ++i)
  {
    int min_index = i;
    for (int j = i + 1; j < nums.size(); ++j)
    {
      if (nums[j] < nums[min_index])
        min_index = j;
    }

    swap(nums[i], nums[min_index]);
  }
}

// 4. Shell Sort
// Time complexity: O(n^2)
// Space complexity: O(1)
void shell_sort(vector<int>& nums)
{
  int h = 1;

  while (h < nums.size() / 3)
    h = (h * 3) + 1; // step: 1 4 13 43 ...

  while (h >= 1)
  {
    for (int i = h; i < nums.size(); ++i)
    {
      for (int j = i; j >= h; j -= h)
      {
        if (nums[j] < nums[j - h])
          swap(nums[j], nums[j - h]);
      }
    }
    h /= 3;
  }
}

int partition_1ways(vector<int>& nums, int l, int r)
{
  int v = nums[l]; // pivot
  int p = l;
  int i = l + 1;
  while (i <= r)
  {
    if (nums[i] < v)
      swap(nums[i], nums[++p]);
    
    i++;
  }
  swap(nums[l], nums[p]);
  return p;
}

void quick_sort_1ways(vector<int>& nums, int l, int r)
{
  if (l >= r)
    return;
  
  int p = partition_1ways(nums, l, r);
  quick_sort_1ways(nums, l, p - 1);
  quick_sort_1ways(nums, p + 1, r);
}


int partition_2ways(vector<int>& nums, int l, int r)
{
  int v = nums[l];
  int i = l + 1;
  int j = r;
  while (true)
  {
    while (i <= r && nums[i] <= v)
      i++;
    while (j >= l && nums[j] > v)
      j--;
    
    if (i > j)
      break;
    
    swap(nums[i++], nums[j--]);
  }
  swap(nums[l], nums[j]);
  return j;
}


void quick_sort_2ways(vector<int>& nums, int l, int r)
{
  if (l >= r)
    return;
  
  int p = partition_2ways(nums, l, r);
  quick_sort_2ways(nums, l, p - 1);
  quick_sort_2ways(nums, p + 1, r);
}

void quick_sort_3ways(vector<int>& nums, int l, int r)
{
  if (l >= r)
    return;
  
  int v = nums[l];
  // nums[l ... lt-1] < v
  // nums[lt ... gt-1] == v
  // nums[gt ... r] > v
  int lt = l;
  int i = l + 1;
  int gt = r + 1;
  while (i < gt)
  {
    if (nums[i] < v)
      swap(nums[i++], nums[++lt]);
    
    else if (nums[i] > v)
      swap(nums[i], nums[--gt]);
    
    else
      i++;
  }

  swap(nums[l], nums[lt]);

  quick_sort_3ways(nums, l, lt - 1);
  quick_sort_3ways(nums, gt, r);
}

// Quick Sort
// Time complexity: O(nlogn)
// Space complexity: O(n)
void quick_sort_1ways(vector<int>& nums)
{
  quick_sort_1ways(nums, 0, nums.size() - 1);
}

// 5. Quick Sort Notes: nice!
// Time complexity: O(nlogn)
// Space complexity: O(n)
void quick_sort_2ways(vector<int>& nums)
{
  quick_sort_2ways(nums, 0, nums.size() - 1);
}

// Quick Sort
// Time complexity: O(nlogn)
// Space complexity: O(n)
void quick_sort_3ways(vector<int>& nums)
{
  quick_sort_3ways(nums, 0, nums.size() - 1);
}

void merge(vector<int>& nums, int l, int mid, int r)
{
  vector<int> aux;
  for (int i = l; i <= r; ++i)
    aux.push_back(nums[i]);
  
  int i = l, j = mid + 1;
  for (int k = l; k <= r; ++k)
  {
    if (i > mid)
    {
      nums[k] = aux[j - l];
      j++;
    }
    
    else if (j > r)
    {
      nums[k] = aux[i - l];
      i++;
    }

    else if (aux[i - l] < aux[j - l])
    {
      nums[k] = aux[i - l];
      i++;
    }
    else
    {
      nums[k] = aux[j - l];
      j++;
    }
  }
}

void merge_sort(vector<int>& nums, int l, int r)
{
  if (l >= r)
    return;
  
  int mid = l + (r - l) / 2;
  merge_sort(nums, l, mid);
  merge_sort(nums, mid + 1, r);

  merge(nums, l, mid, r);
}

// 6. Merge Sort
// Time complexity: O(nlogn)
// Space complexity: O(n)
void merge_sort(vector<int>& nums)
{
  merge_sort(nums, 0, nums.size() - 1);
}

void shift_down(vector<int>& nums, int size, int k)
{
  while (2 * k + 1 < size)
  {
    int j = 2 * k + 1; // this is left child index.
    if (j + 1 < size && nums[j+1] > nums[j])
      j++;
    
    if (nums[k] < nums[j])
      swap(nums[k], nums[j]);
    
    k = j; // continue to sink.
  }
}

// 7. Heap Sort
// Time complexity: O(nlogn)
// Space complexity: O(1)
void heap_sort(vector<int>& nums)
{
  // The last non-leaf node starts to sink.
  for (int i = (nums.size() - 2) / 2; i >= 0; --i)
    shift_down(nums, nums.size(), i);
  
  for (int i = nums.size() - 1; i > 0; --i)
  {
    swap(nums[0], nums[i]);
    shift_down(nums, i, 0);
  }
}

// 8. Counting Sort // NOTES: 0 <= nums[i] < 100
// Time complexity: O(n)
// Space complexity: O(n)
void counting_sort(vector<int>& nums)
{
  if (nums.size() < 2)
    return;

  vector<int> count(100, 0);
  // Record the frequency of each element
  for (int i = 0; i < nums.size(); ++i)
    count[nums[i]]++;

  for (int i = 1; i < 100; ++i)
    count[i] += count[i - 1];
  
  vector<int> aux(nums.size(), 0);
  for (int i = nums.size() - 1; i >= 0; --i)
  {
    aux[count[nums[i]] - 1] = nums[i];
    count[nums[i]]--;
  }

  nums = aux;
}

// 9. Radix Sort

int get_max(vector<int>& nums)
{
  int max_val = nums[0];
  for (int i = 1; i < nums.size(); ++i)
  {
    if (nums[i] > max_val)
      max_val = nums[i];
  }

  return max_val;
}


void counting_sort_ten(vector<int>& nums, int place)
{
  int max = 10; // [0 ..9]
  vector<int> count(10, 0);
  for (int i = 0; i < nums.size(); ++i)
    count[(nums[i] / place) % 10]++;

  for (int i = 1; i < max; ++i)
    count[i] += count[i - 1];

  vector<int> aux(nums.size(), 0);
  for (int i = nums.size() - 1; i >= 0; --i)
  {
    aux[count[(nums[i] / place) % 10] - 1] = nums[i];
    count[(nums[i] / place) % 10]--;
  }

  nums = aux;
}

// 9. Radix Sort
// Time complexity: O(n)
// Space comnplexity: O(1)
void radix_sort(vector<int>& nums)
{
  if (nums.size() < 2)
    return;

  int max_val = get_max(nums);
  for (int place = 1; max_val / place > 0; place *= 10)
    counting_sort_ten(nums, place);
}

// 10. Bucket Sort // NOTES: 0 <= nums[i] < 100
void bucket_sort(vector<int>& nums)
{
  vector<vector<int>> bucket(10);

  for (int i = 0; i < nums.size(); ++i)
    bucket[nums[i] / 10].push_back(nums[i]);

  int k = 0;
  for (int i = 0; i < 10; ++i)
  {
    for (int j = 0; j < bucket[i].size(); ++j)
      nums[k++] = bucket[i][j];
  }
  // An almost ordered array, called insertion sort.
  insertion_sort(nums);
}

#endif // SORT_ALGO_H
```

