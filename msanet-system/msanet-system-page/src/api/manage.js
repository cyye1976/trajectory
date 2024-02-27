import Vue from 'vue'
import axios from "@/api/request1";


//post
export function postAction(url,parameter) {
  return axios({
    url: url,
    method:'post' ,
    data: parameter
  })
}

//post method= {post | put}
export function httpAction(url,parameter,method) {
  return axios({
    url: url,
    method:method ,
    data: parameter
  })
}

//put
export function putAction(url,parameter) {
  return axios({
    url: url,
    method:'put',
    data: parameter
  })
}

//get
export function getAction(url) {
  return axios({
    url: url,
    method: 'get'
  })
}

//get
export function getActionBlob(url) {
  return axios({
    url: url,
    method: 'get',
    responseType: 'blob'
  })
}

//post
export function postActionBlob(url, parameter) {
  return axios({
    url: url,
    method: 'post',
    data: parameter,
    responseType: 'blob'
  })
}

//deleteAction
export function deleteAction(url,parameter) {
  return axios({
    url: url,
    method: 'delete',
    params: parameter
  })
}
