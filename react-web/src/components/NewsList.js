import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import List from '@material-ui/core/List';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import ListSubheader from '@material-ui/core/ListSubheader';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import { FixedSizeList } from 'react-window';
import Typography from '@material-ui/core/Typography';
import { Button } from '@material-ui/core';
import NewsCard from './NewsCard';
import CircularProgress from '@material-ui/core/CircularProgress';



const useStyles = makeStyles((theme) => ({
  root: {
  },
  listSubHeader: {
    textAlign: 'initial'
  },
  listTitle: {
    margin: theme.spacing(1, 1, 1),
    display: "inline-flex"
  },
  saveButton: {
    marginLeft: "auto",
    height: "36px",
    width: "64px"
  },
  emptyText: {
    margin: theme.spacing(2, 0, 2),
    textAlign: 'center'
  },
  buttonProgress: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    marginTop: -12,
    marginLeft: -12,
  }
}));


function NewsList(props) {
  const classes = useStyles();

  return (
    <Box className={classes.root} component={Grid} container direction="column">
      {props.newsData.map((news) => (
        <NewsCard 
          key={news.id} 
          date={news.date}
          title={news.title}
          companyName={news.companyName}
          paragraph={news.paragraph}
        >
        </NewsCard>
      ))}
    </Box>
  );
}

export default NewsList